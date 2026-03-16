from dataclasses import dataclass
import jax
from typing import List

# Time function kinds (compiled into arrays; evaluated inside the JITted step function)
TIME_FN_CONSTANT = 0
TIME_FN_RICKER = 1

TIME_FN_NPARAM = 4  # [p0, p1, p2, p3] meaning depends on kind

@dataclass
class NodalLoadData:
    field_name: str
    node_indices: jax.Array   # (n_loaded,) field-local
    force_pattern: jax.Array  # (n_loaded, ncomp)
    time_fn_id: int           # index into time function table

@dataclass
class BodyLoadData:
    pass

@dataclass
class SurfaceLoadData:
    pass

@dataclass
class LoadPlan:
    nodal_loads: List[NodalLoadData]
    body_loads: List[BodyLoadData]
    surface_loads: List[SurfaceLoadData]

    # Time function table
    # kinds:  (n_time_fns,) int32
    # params: (n_time_fns, TIME_FN_NPARAM) float32
    time_fn_kinds: jax.Array
    time_fn_params: jax.Array
