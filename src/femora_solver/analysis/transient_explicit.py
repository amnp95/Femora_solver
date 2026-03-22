import jax
import jax.numpy as jnp
from femora_solver.state.state import State
from femora_solver.compile.execution_plan import ExecutionPlan


def _eval_time_fn(kind: jax.Array, params: jax.Array, t: jax.Array) -> jax.Array:
    """Return scalar load scale factor at time t.

    kind meanings (see femora_solver.loads.load_plan):
      0: Constant   -> amp
      1: Linear     -> ramp from t0 to t1 up to amp
      2: TimeSeries -> interpolated user data scaled by amp (shifted by t_offset)
    """
    kind_i = kind.astype(jnp.int32)

    def _constant(_):
        return params[0] # amp

    def _linear(_):
        t0  = params[0]
        t1  = params[1]
        amp = params[2]
        
        # Calculate normalized slope (0 to 1) between t0 and t1
        # m = (t - t0) / (t1 - t0)
        m = (t - t0) / (t1 - t0)
        m = jnp.clip(m, 0.0, 1.0)
        return amp * m

    def _time_series(_):
        # TODO: Implement 1D Linear Interpolation for TimeSeries data.
        # For now, just return the constant amplitude (multiplier).
        series_id = params[0]
        amp       = params[1]
        t_offset  = params[2]
        return amp

    return jax.lax.switch(kind_i, (_constant, _linear, _time_series), operand=None)

def build_step_fn(plan: ExecutionPlan, dt: float):
    def step_fn(state: State, t: float):
        # Explicit Dynamics (Central Difference / Newmark Beta=0, Gamma=0.5)
        # Input state contains u_n, v_n, a_n
        
        # 1. Compute u_{n+1} = u_n + dt * v_n + 0.5 * dt^2 * a_n
        u_pred = {}
        for name, field in state.fields.items():
            u_new = field.u + dt * field.v + 0.5 * (dt ** 2) * field.a
            
            # Apply Dirichlet to u_{n+1}
            if name in plan.constraint_plan.dirichlet:
                d_data = plan.constraint_plan.dirichlet[name]
                flat_u = u_new.reshape(-1)
                flat_u = flat_u.at[d_data.flat_indices].set(d_data.values)
                u_new = flat_u.reshape(u_new.shape)
            
            u_pred[name] = u_new

        # 2. Ghost exchange (Phase 1: skip)
        
        # 3. Compute internal forces F_int(u_{n+1})
        f_int = {name: jnp.zeros_like(state.fields[name].u) for name in state.fields}
        new_history = state.history.copy()
        
        for block in plan.element_blocks:
            f_block, hist_block = block.compute_internal_forces(
                u_pred, 
                state.history.get(block.block_id)
            )
            for k, v in f_block.items():
                f_int[k] = f_int[k] + v
            if hist_block is not None:
                new_history[block.block_id] = hist_block
                
        # 4. Compute external forces
        f_ext = {name: jnp.zeros_like(state.fields[name].u) for name in state.fields}
        for load in plan.load_plan.nodal_loads:
            kind = plan.load_plan.time_fn_kinds[load.time_fn_id]
            params = plan.load_plan.time_fn_params[load.time_fn_id]
            scale = _eval_time_fn(kind, params, t)
            f_ext[load.field_name] = f_ext[load.field_name].at[load.node_indices].add(load.force_pattern * scale)
            
        # 5. Compute a_{n+1} = M^-1 (F_ext - F_int)
        new_fields = {}
        for name, field in state.fields.items():
            m = field.m
            # Avoid division by zero for fixed/massless DOFs
            m_safe = jnp.where(m == 0, 1.0, m)
            a_new = (f_ext[name] - f_int[name]) / m_safe
            
            # Apply Dirichlet to a_{n+1}
            if name in plan.constraint_plan.dirichlet:
                d_data = plan.constraint_plan.dirichlet[name]
                flat_a = a_new.reshape(-1)
                flat_a = flat_a.at[d_data.flat_indices].set(0.0)
                a_new = flat_a.reshape(a_new.shape)
                
            # 6. Compute v_{n+1} = v_n + 0.5 * dt * (a_n + a_{n+1})
            v_new = field.v + 0.5 * dt * (field.a + a_new)
            
            # Apply Dirichlet to v_{n+1}
            if name in plan.constraint_plan.dirichlet:
                flat_v = v_new.reshape(-1)
                flat_v = flat_v.at[d_data.flat_indices].set(0.0)
                v_new = flat_v.reshape(v_new.shape)
                
            new_fields[name] = field.replace(u=u_pred[name], v=v_new, a=a_new)

        # 7. Sample recorder payload
        recorder_payload = {}
        for rec in plan.recorder_plan.recorders:
            # Gather requested nodes
            u_field = new_fields[rec.field_name].u
            rec_data = u_field[rec.node_indices]
            recorder_payload[rec.recorder_id] = rec_data

        new_state = state.replace(
            time=t,
            step=state.step + 1,
            fields=new_fields,
            history=new_history
        )
        return new_state, recorder_payload

    return step_fn


# ======================================================================
# Profiled step: decomposed into individually-timed phases
# ======================================================================

def build_profiled_step_phases(plan: ExecutionPlan, dt: float):
    """Build individual phase callables for profiled execution.

    Returns a dict of phase_name -> callable(state, t) -> result.
    Each phase is designed to be called in sequence with
    jax.block_until_ready() barriers between them.

    The runner calls these in order:
        1. predict_displacement(state, t)        -> u_pred
        2. bulk_element_kernels(u_pred, state)   -> f_int, new_history
        3. interface_coupling_kernels(...)        -> f_coupling (placeholder)
        4. compute_external_forces(state, t)      -> f_ext
        5. constraint_and_update(state, u_pred, f_int, f_ext, t) -> new_state
        6. recorder_extraction(new_state)         -> recorder_payload
    """

    # --- Phase 1: Predict displacement + Dirichlet on u ---
    @jax.jit
    def predict_displacement(state: State, t):
        u_pred = {}
        for name, field in state.fields.items():
            u_new = field.u + dt * field.v + 0.5 * (dt ** 2) * field.a
            if name in plan.constraint_plan.dirichlet:
                d_data = plan.constraint_plan.dirichlet[name]
                flat_u = u_new.reshape(-1)
                flat_u = flat_u.at[d_data.flat_indices].set(d_data.values)
                u_new = flat_u.reshape(u_new.shape)
            u_pred[name] = u_new
        return u_pred

    # --- Phase 2: Bulk element kernels ---
    @jax.jit
    def bulk_element_kernels(u_pred, history):
        f_int = {name: jnp.zeros_like(v) for name, v in u_pred.items()}
        new_history = history.copy()
        for block in plan.element_blocks:
            f_block, hist_block = block.compute_internal_forces(
                u_pred,
                history.get(block.block_id)
            )
            for k, v in f_block.items():
                f_int[k] = f_int[k] + v
            if hist_block is not None:
                new_history[block.block_id] = hist_block
        return f_int, new_history

    # --- Phase 3: Interface/coupling kernels (placeholder) ---
    @jax.jit
    def interface_coupling_kernels(f_int):
        # Phase 1: no couplings. Returns f_int unchanged.
        return f_int

    # --- Phase 4: External forces ---
    @jax.jit
    def compute_external_forces(fields, t):
        f_ext = {name: jnp.zeros_like(field.u) for name, field in fields.items()}
        for load in plan.load_plan.nodal_loads:
            kind = plan.load_plan.time_fn_kinds[load.time_fn_id]
            params = plan.load_plan.time_fn_params[load.time_fn_id]
            scale = _eval_time_fn(kind, params, t)
            f_ext[load.field_name] = f_ext[load.field_name].at[load.node_indices].add(
                load.force_pattern * scale
            )
        return f_ext

    # --- Phase 5: Constraint enforcement + explicit update ---
    @jax.jit
    def constraint_and_update(state, u_pred, f_int, f_ext, t):
        new_fields = {}
        for name, field in state.fields.items():
            m = field.m
            m_safe = jnp.where(m == 0, 1.0, m)
            a_new = (f_ext[name] - f_int[name]) / m_safe

            if name in plan.constraint_plan.dirichlet:
                d_data = plan.constraint_plan.dirichlet[name]
                flat_a = a_new.reshape(-1)
                flat_a = flat_a.at[d_data.flat_indices].set(0.0)
                a_new = flat_a.reshape(a_new.shape)

            v_new = field.v + 0.5 * dt * (field.a + a_new)

            if name in plan.constraint_plan.dirichlet:
                flat_v = v_new.reshape(-1)
                flat_v = flat_v.at[d_data.flat_indices].set(0.0)
                v_new = flat_v.reshape(v_new.shape)

            new_fields[name] = field.replace(u=u_pred[name], v=v_new, a=a_new)

        return state.replace(
            time=t,
            step=state.step + 1,
            fields=new_fields,
        )

    # --- Phase 6: Recorder extraction ---
    @jax.jit
    def recorder_extraction(state):
        payload = {}
        for rec in plan.recorder_plan.recorders:
            u_field = state.fields[rec.field_name].u
            payload[rec.recorder_id] = u_field[rec.node_indices]
        return payload

    return {
        "predict_displacement": predict_displacement,
        "bulk_element_kernels": bulk_element_kernels,
        "interface_coupling_kernels": interface_coupling_kernels,
        "compute_external_forces": compute_external_forces,
        "constraint_and_update": constraint_and_update,
        "recorder_extraction": recorder_extraction,
    }
