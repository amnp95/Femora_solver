import jax
import jax.numpy as jnp
from femora_solver.compile.execution_plan import ExecutionPlan
from femora_solver.elements.families.hex8 import Hex8Block
from femora_solver.loads.load_plan import (
    LoadPlan,
    NodalLoadData,
    TIME_FN_CONSTANT,
    TIME_FN_RICKER,
    TIME_FN_NPARAM,
)
from femora_solver.constraints.constraint_plan import ConstraintPlan, DirichletData
from femora_solver.recorder.recorder_plan import RecorderPlan, RecorderEntry
from femora_solver.compile.comm_plan import CommPlan
from femora_solver.fields.field_space import FieldSpace
from femora_solver.state.state import State, FieldState

from femora_solver.elements.families.beam2 import Beam2Block

class Compiler:
    def full_compile(self, model: "Model", profiler=None) -> ExecutionPlan:
        # -----------------------------------------------------------
        # 1. Field layout construction
        # -----------------------------------------------------------
        if profiler:
            profiler.mark_start()

        with (profiler.region("field_layout_construction") if profiler else _nullctx()):
            required_fields = {"U"} # Always U
            for b_data in model._blocks.values():
                if b_data["family"] == "Beam2":
                    required_fields.add("R")
            
            fields = {}
            if "U" in required_fields:
                fields["U"] = FieldSpace("U", 3)
            if "R" in required_fields:
                fields["R"] = FieldSpace("R", 3)
        
        # -----------------------------------------------------------
        # 2. Build Element Blocks
        # -----------------------------------------------------------
        with (profiler.region("model_compilation") if profiler else _nullctx()):
            element_blocks = []
            for b_id, b_data in model._blocks.items():
                if b_data["family"] == "Hex8":
                    coords = jnp.array(b_data["coords"])
                    conn = jnp.array(b_data["conn"])
                    params = jnp.array(b_data["params"])
                    
                    block = Hex8Block(
                        block_id=b_id,
                        family="Hex8",
                        field_names=["U"],
                        connectivity=conn,
                        material_params=params,
                        history_shape=(conn.shape[0], 8, 0),
                        coords=coords
                    )
                    element_blocks.append(block)
                elif b_data["family"] == "Beam2":
                    coords = jnp.array(b_data["coords"])
                    conn = jnp.array(b_data["conn"])
                    params = jnp.array(b_data["params"])
                    sec_params = jnp.array(b_data["section_params"])
                    
                    block = Beam2Block(
                        block_id=b_id,
                        family="Beam2",
                        field_names=["U", "R"],
                        connectivity=conn,
                        material_params=params,
                        history_shape=(conn.shape[0], 0),
                        coords=coords,
                        section_params=sec_params
                    )
                    element_blocks.append(block)

            # -----------------------------------------------------------
            # 3. Constraints
            # -----------------------------------------------------------
            dirichlet = {}
            for c in model._constraints:
                if c["kind"] == "SPC":
                    f_name = c.get("field", "U")
                    nodes = jnp.array(c["node_indices"])
                    ncomp = fields[f_name].ncomp
                    
                    flat_idx = []
                    for n in nodes:
                        for d in c["components"]:
                            flat_idx.append(n * ncomp + d)
                    
                    if f_name not in dirichlet:
                        dirichlet[f_name] = DirichletData(
                            flat_indices=jnp.array(flat_idx),
                            values=jnp.zeros(len(flat_idx))
                        )
                    else:
                        # Append to existing
                        existing = dirichlet[f_name]
                        new_indices = jnp.concatenate([existing.flat_indices, jnp.array(flat_idx)])
                        new_values = jnp.concatenate([existing.values, jnp.zeros(len(flat_idx))])
                        dirichlet[f_name] = DirichletData(flat_indices=new_indices, values=new_values)

            c_plan = ConstraintPlan(dirichlet=dirichlet)
            
            # -----------------------------------------------------------
            # 4. Loads + time functions
            # -----------------------------------------------------------
            # Table index 0 is always Constant(scale=1) for backward compatibility.
            time_fn_entries = [(TIME_FN_CONSTANT, (1.0, 0.0, 0.0, 0.0))]
            time_fn_map = {time_fn_entries[0]: 0}

            nodal_loads = []
            for l in model._loads:
                if l["kind"] != "Nodal":
                    continue

                node_indices = jnp.array(l["node_indices"], dtype=jnp.int32)
                force_pattern = jnp.array(l["force"], dtype=jnp.float32)

                # Allow specifying a single vector for many nodes (broadcast it).
                if force_pattern.ndim == 1:
                    force_pattern = jnp.broadcast_to(force_pattern, (node_indices.shape[0], force_pattern.shape[0]))
                elif force_pattern.ndim == 2 and force_pattern.shape[0] == 1 and node_indices.shape[0] != 1:
                    force_pattern = jnp.broadcast_to(force_pattern, (node_indices.shape[0], force_pattern.shape[1]))

                tf_key = l.get("time_fn")
                if tf_key not in time_fn_map:
                    time_fn_map[tf_key] = len(time_fn_entries)
                    time_fn_entries.append(tf_key)
                time_fn_id = int(time_fn_map[tf_key])

                nodal_loads.append(
                    NodalLoadData(
                        field_name=l["field"],
                        node_indices=node_indices,
                        force_pattern=force_pattern,
                        time_fn_id=time_fn_id,
                    )
                )

            # Pack time functions into arrays
            kinds = jnp.array([k for (k, _p) in time_fn_entries], dtype=jnp.int32)
            params = jnp.array([list(p) for (_k, p) in time_fn_entries], dtype=jnp.float32)
            if params.shape[1] != TIME_FN_NPARAM:
                raise RuntimeError("Internal error: time function param width mismatch.")

            l_plan = LoadPlan(
                nodal_loads=nodal_loads,
                body_loads=[],
                surface_loads=[],
                time_fn_kinds=kinds,
                time_fn_params=params,
            )
            
            # -----------------------------------------------------------
            # 5. Recorders
            # -----------------------------------------------------------
            recorders = []
            for r in model._recorders:
                if r["kind"] == "NodeSet":
                    recorders.append(RecorderEntry(
                        recorder_id=r["id"],
                        field_name=r["field"],
                        node_indices=jnp.array(r["node_indices"]),
                        component_mask=jnp.ones(fields[r["field"]].ncomp, dtype=bool),
                        interval=r["interval"],
                        output_path=r["file"]
                    ))
            r_plan = RecorderPlan(recorders=recorders, chunk_size=model._chunk_size)

        # -----------------------------------------------------------
        # 6. Partitioning / CommPlan
        # -----------------------------------------------------------
        with (profiler.region("partitioning") if profiler else _nullctx()):
            comm_plan = CommPlan(n_ranks=1, this_rank=0, fields={})

        # -----------------------------------------------------------
        # 7. Mass assembly
        # -----------------------------------------------------------
        with (profiler.region("model_compilation") if profiler else _nullctx()):
            mass_arrays = {f_name: jnp.zeros((model._num_nodes, field.ncomp)) for f_name, field in fields.items()}
            for block in element_blocks:
                if hasattr(block, "compute_lumped_mass"):
                    block_mass = block.compute_lumped_mass()
                    for f_name, (indices, values) in block_mass.items():
                        if f_name in mass_arrays:
                            m_field = mass_arrays[f_name]
                            m_field = m_field.at[indices].add(values)
                            mass_arrays[f_name] = m_field

        plan = ExecutionPlan(
            fields=fields,
            element_blocks=element_blocks,
            constraint_plan=c_plan,
            coupling_plan=None,
            load_plan=l_plan,
            recorder_plan=r_plan,
            comm_plan=comm_plan,
            mass_arrays=mass_arrays,
            sharding={},
            chunk_size=model._chunk_size
        )
        return plan


# ------------------------------------------------------------------
# Utility: no-op context manager when profiler is None
# ------------------------------------------------------------------
from contextlib import contextmanager as _contextmanager

@_contextmanager
def _nullctx():
    yield
