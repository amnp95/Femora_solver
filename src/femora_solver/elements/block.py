from dataclasses import dataclass
import jax
from typing import Dict, List, Any
from femora_solver.fields.field_space import FieldSpace

@dataclass
class ElementBlock:
    block_id: str
    family: str                          # "Hex8", "Beam2", etc.
    field_names: List[str]               # e.g. ["U"] or ["U", "R"]
    connectivity: jax.Array              # (n_elem, n_nodes_per_elem) field-local
    material_params: jax.Array           # (n_elem, n_params) or (n_params,) if uniform
    history_shape: tuple                 # (n_elem, n_qp, n_hist_vars)
    # quadrature_rule: Any # Add quadrature rule later

    def compute_internal_forces(
        self,
        field_u: Dict[str, jax.Array],  # field_name -> (n_nodes, ncomp)
        history: jax.Array,
    ) -> tuple[Dict[str, jax.Array], jax.Array]:
        # returns (force_contributions per field, updated_history)
        raise NotImplementedError

    def compute_stable_dt(
        self,
        field_u: Dict[str, jax.Array],
        material_params: jax.Array,
    ) -> float:
        raise NotImplementedError
