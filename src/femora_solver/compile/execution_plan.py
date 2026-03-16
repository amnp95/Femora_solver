from dataclasses import dataclass
from typing import Dict, List, Any
import jax
from femora_solver.fields.field_space import FieldSpace
from femora_solver.elements.block import ElementBlock
from femora_solver.constraints.constraint_plan import ConstraintPlan
from femora_solver.loads.load_plan import LoadPlan
from femora_solver.recorder.recorder_plan import RecorderPlan
from femora_solver.compile.comm_plan import CommPlan

# Placeholder for CouplingPlan
@dataclass
class CouplingPlan:
    pass

@dataclass
class ExecutionPlan:
    fields: Dict[str, FieldSpace]
    element_blocks: List[ElementBlock]
    constraint_plan: ConstraintPlan
    coupling_plan: CouplingPlan
    load_plan: LoadPlan
    recorder_plan: RecorderPlan
    comm_plan: CommPlan
    mass_arrays: Dict[str, jax.Array]    # field_name -> (n_owned, ncomp)
    sharding: Dict[str, Any]             # dict[str, jax.sharding.Sharding]
    chunk_size: int
