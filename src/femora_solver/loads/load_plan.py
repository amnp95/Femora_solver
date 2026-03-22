from dataclasses import dataclass
import jax
from typing import List

# Time function kinds (compiled into arrays; evaluated inside the JITted step function)
TIME_FN_CONSTANT = 0
TIME_FN_LINEAR = 1
TIME_FN_TIME_SERIES = 2

TIME_FN_NPARAM = 4  # [p0, p1, p2, p3] meaning depends on kind

def normalize_time_fn(spec):
    """
    Normalizes time function specification into (kind: int, params: tuple[float,float,float,float])
    
    This provides immediate validation for user-provided time functions.
    Supported Kinds:
        - Constant: (scale)
        - Linear:   (t0, t1, amp)
        - TimeSeries: (path_id) -- custom user data
    """
    if spec is None:
        return (TIME_FN_CONSTANT, (1.0, 0.0, 0.0, 0.0))
    if isinstance(spec, (int, float)):
        return (TIME_FN_CONSTANT, (float(spec), 0.0, 0.0, 0.0))
    if isinstance(spec, dict):
        kind = str(spec.get("kind", "Constant")).strip().lower()
        
        # 0. Constant
        if kind in ("const", "constant"):
            scale = float(spec.get("scale", spec.get("value", 1.0)))
            return (TIME_FN_CONSTANT, (scale, 0.0, 0.0, 0.0))
            
        # 1. Linear Ramp
        if kind in ("lin", "linear", "ramp"):
            t0 = float(spec.get("t0", spec.get("start_time", 0.0)))
            t1 = float(spec.get("t1", spec.get("end_time", 1.0)))
            amp = float(spec.get("amp", spec.get("scale", 1.0)))
            if t1 <= t0:
                raise ValueError(f"Linear time_fn error: t1 ({t1}) must be greater than t0 ({t0}).")
            return (TIME_FN_LINEAR, (t0, t1, amp, 0.0))
            
        # 2. Time Series (Experimental Data / Any arbitrary path)
        if kind in ("path", "timeseries", "time_series"):
            # Note: The raw list of [t, v] points lives in a separate data pool.
            # Here, we only store the "Series ID" if it was pre-allocated, 
            # OR we just flag it for the compiler to handle later.
            series_id = float(spec.get("id", spec.get("series_id", 0.0)))
            return (TIME_FN_TIME_SERIES, (series_id, 0.0, 0.0, 0.0))
            
        raise ValueError(f"Unsupported time_fn kind: {spec.get('kind')!r}")
    
    raise TypeError(
        "time_fn must be None, a number (constant scale), or a dict like {kind: 'Linear', t0: ..., t1: ..., amp: ...}."
    )

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
