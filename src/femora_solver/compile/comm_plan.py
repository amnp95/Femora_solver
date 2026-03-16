import numpy as np
from dataclasses import dataclass
import jax

@dataclass
class FieldCommPlan:
    field_name: str
    ncomp: int
    export_rank: np.ndarray     # (n_exports,) CPU-side for MPI dispatch
    export_indices: np.ndarray  # (n_exports,) field-local owned indices
    import_rank: np.ndarray     # (n_imports,)
    import_indices: np.ndarray  # (n_imports,) field-local ghost indices
    pack_perm: jax.Array        # (n_exports,) for jnp.take
    unpack_perm: jax.Array      # (n_imports,) for .at[].set()

@dataclass
class CommPlan:
    n_ranks: int
    this_rank: int
    fields: dict[str, FieldCommPlan]
