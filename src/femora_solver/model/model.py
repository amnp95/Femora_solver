import jax.numpy as jnp
from femora_solver.compile.compiler import Compiler
from femora_solver.analysis.runner import Runner
from femora_solver.state.state import State, FieldState
from typing import Optional

class Model:
    def __init__(self, name: str):
        self.name = name
        self._nodes = None
        self._num_nodes = 0
        self._blocks = {}
        self._materials = {}
        self._constraints = []
        self._loads = []
        self._recorders = []
        self._chunk_size = 10
        
        self._compiler = Compiler()
        self._runner = Runner()
        self._execution_plan = None
        self._state = None
        self._dirty = 4 # FULL compile
        
    def add_nodes(self, coords):
        self._nodes = jnp.array(coords)
        self._num_nodes = len(coords)
        self._dirty = 4
        
    def add_elements(self, conn, family="Hex8", material=None, block_id=None, section_params=None):
        if block_id is None:
            block_id = f"block_{len(self._blocks)}"
        
        conn_array = jnp.array(conn)
        coords_array = self._nodes[conn_array]
        
        mat = self._materials[material]
        params = jnp.array([mat["E"], mat["nu"], mat.get("rho", 0.0)])
        
        self._blocks[block_id] = {
            "family": family,
            "conn": conn,
            "coords": coords_array,
            "params": params,
            "section_params": section_params
        }
        self._dirty = 4
        
    def add_material(self, name, kind, **kwargs):
        self._materials[name] = {"kind": kind, **kwargs}
        self._dirty = 4
        
    def add_constraint(self, name, kind, node_indices, dofs=None, components=None, field="U"):
        if components is None:
            components = [0, 1, 2] # Default x,y,z
        self._constraints.append({
            "name": name,
            "kind": kind,
            "node_indices": node_indices,
            "components": components,
            "field": field
        })
        self._dirty = max(self._dirty, 2)
        
    def add_load(self, name, kind, field, node_indices, force, time_fn=None):
        self._loads.append({
            "name": name,
            "kind": kind,
            "field": field,
            "node_indices": node_indices,
            "force": force,
            "time_fn": time_fn,
        })
        self._dirty = max(self._dirty, 1)
        
    def add_recorder(self, name, kind, node_indices, field, interval, file):
        self._recorders.append({
            "id": name,
            "kind": kind,
            "node_indices": node_indices,
            "field": field,
            "interval": interval,
            "file": file
        })
        self._dirty = max(self._dirty, 1)
        
    def _init_state(self):
        # Create initial state
        n_nodes = self._num_nodes
        
        fields = {}
        for name, field_space in self._execution_plan.fields.items():
            ncomp = field_space.ncomp
            u = jnp.zeros((n_nodes, ncomp))
            v = jnp.zeros((n_nodes, ncomp))
            a = jnp.zeros((n_nodes, ncomp))
            m = self._execution_plan.mass_arrays[name]
            
            field_state = FieldState(
                name=name, ncomp=ncomp,
                u=u, v=v, a=a, m=m,
                n_owned=n_nodes, n_ghost=0
            )
            fields[name] = field_state
        
        self._state = State(
            time=0.0,
            step=0,
            fields=fields,
            history={}
        )

    def run(self, dt, time, progress: bool = False, progress_every: int = 1, sync_progress: Optional[bool] = None):
        if self._dirty >= 4:
            self._execution_plan = self._compiler.full_compile(self)
            if self._state is None:
                self._init_state()
        self._dirty = 0
        
        self._state = self._runner.run(
            self._execution_plan,
            self._state,
            dt,
            time,
            progress=progress,
            progress_every=progress_every,
            sync_progress=sync_progress,
        )
