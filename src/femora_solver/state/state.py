import dataclasses
from dataclasses import dataclass
import jax
from typing import Dict, Any, Optional

@jax.tree_util.register_pytree_node_class
@dataclass
class FieldState:
    name: str
    ncomp: int
    u: jax.Array    # (n_nodes, ncomp) displacement/field value
    v: jax.Array    # (n_nodes, ncomp) velocity
    a: jax.Array    # (n_nodes, ncomp) acceleration
    m: jax.Array    # (n_nodes, ncomp) lumped mass (per component)
    # owned/ghost split
    n_owned: int
    n_ghost: int

    @property
    def n_total(self) -> int:
        return self.n_owned + self.n_ghost
        
    def tree_flatten(self):
        children = (self.u, self.v, self.a, self.m)
        aux_data = (self.name, self.ncomp, self.n_owned, self.n_ghost)
        return (children, aux_data)
        
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        u, v, a, m = children
        name, ncomp, n_owned, n_ghost = aux_data
        return cls(name, ncomp, u, v, a, m, n_owned, n_ghost)

    def replace(self, **kwargs) -> "FieldState":
        return dataclasses.replace(self, **kwargs)

@jax.tree_util.register_pytree_node_class
@dataclass
class State:
    time: float
    step: int
    fields: Dict[str, FieldState]        # keyed by field name
    history: Dict[str, jax.Array]        # keyed by block id
    energy_kinetic: Optional[float] = None
    energy_internal: Optional[float] = None

    def tree_flatten(self):
        children = (self.time, self.step, self.fields, self.history, self.energy_kinetic, self.energy_internal)
        aux_data = None
        return (children, aux_data)
        
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        time, step, fields, history, energy_kinetic, energy_internal = children
        return cls(time, step, fields, history, energy_kinetic, energy_internal)

    def replace(self, **kwargs) -> "State":
        return dataclasses.replace(self, **kwargs)
