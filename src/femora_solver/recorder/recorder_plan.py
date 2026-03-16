from dataclasses import dataclass
import jax
from typing import List

@dataclass
class RecorderEntry:
    recorder_id: str
    field_name: str
    node_indices: jax.Array     # (n_rec,) field-local
    component_mask: jax.Array   # (ncomp,) bool
    interval: int               # every N steps
    output_path: str

@dataclass
class RecorderPlan:
    recorders: List[RecorderEntry]
    chunk_size: int
