import jax
import jax.numpy as jnp
from femora_solver.state.state import State
from femora_solver.compile.execution_plan import ExecutionPlan


def _eval_time_fn(kind: jax.Array, params: jax.Array, t: jax.Array) -> jax.Array:
    """Return scalar load scale factor at time t.

    kind meanings (see femora_solver.loads.load_plan):
      0: Constant -> params[0]
      1: Ricker   -> amp*(1-2a)*exp(-a), a=(pi*f0*(t-t0))^2
    """
    kind_i = kind.astype(jnp.int32)

    def _constant(_):
        return params[0]

    def _ricker(_):
        amp = params[0]
        f0 = params[1]
        t0 = params[2]
        x = jnp.pi * f0 * (t - t0)
        a = x * x
        return amp * (1.0 - 2.0 * a) * jnp.exp(-a)

    return jax.lax.switch(kind_i, (_constant, _ricker), operand=None)

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
