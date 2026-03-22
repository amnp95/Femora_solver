from dataclasses import dataclass
import jax
from typing import List

# Time function kinds (compiled into arrays; evaluated inside the JITted step function)
TIME_FN_CONSTANT = 0
TIME_FN_RICKER = 1

TIME_FN_NPARAM = 4  # [p0, p1, p2, p3] meaning depends on kind

def normalize_time_fn(spec):
    """
    Normalizes time function specification into (kind: int, params: tuple[float,float,float,float])
    
    This provides immediate validation for user-provided time functions.
    """
    from typing import Dict, Any, Union
    
    if spec is None:
        return (TIME_FN_CONSTANT, (1.0, 0.0, 0.0, 0.0))
    if isinstance(spec, (int, float)):
        return (TIME_FN_CONSTANT, (float(spec), 0.0, 0.0, 0.0))
    if isinstance(spec, dict):
        kind = str(spec.get("kind", "Constant")).strip().lower()
        if kind in ("const", "constant"):
            scale = spec.get("scale", spec.get("value", 1.0))
            return (TIME_FN_CONSTANT, (float(scale), 0.0, 0.0, 0.0))
        if kind in ("ricker", "ricker_wavelet", "rickerwavelet"):
            amp = float(spec.get("amp", spec.get("scale", 1.0)))
            f0 = spec.get("f0", spec.get("freq", spec.get("frequency")))
            if f0 is None:
                raise ValueError("Ricker time_fn requires 'f0' (Hz).")
            t0 = float(spec.get("t0", spec.get("t_peak", 0.0)))
            return (TIME_FN_RICKER, (amp, float(f0), t0, 0.0))
        raise ValueError(f"Unsupported time_fn kind: {spec.get('kind')!r}")
    
    raise TypeError(
        "time_fn must be None, a number (constant scale), or a dict like {kind: 'Ricker', f0: ..., t0: ...}."
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
