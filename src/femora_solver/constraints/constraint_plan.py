from dataclasses import dataclass
import jax
from typing import Dict, Union

@dataclass
class DirichletData:
    flat_indices: jax.Array   # (n_fixed,) — flat index into field u array
    values: jax.Array         # (n_fixed,) — prescribed value (static)

@dataclass
class ConstraintPlan:
    # Dirichlet BCs per field
    dirichlet: Dict[str, DirichletData]  # field_name -> DirichletData

    # MPC (general linear) - optional for Phase 1
    mpc_rows: Union[jax.Array, None] = None     # (n_mpc_terms,)
    mpc_cols: Union[jax.Array, None] = None     # (n_mpc_terms,)
    mpc_coeffs: Union[jax.Array, None] = None   # (n_mpc_terms,)
    mpc_rhs: Union[jax.Array, None] = None      # (n_mpc,)
