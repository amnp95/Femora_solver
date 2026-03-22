import jax.numpy as jnp
from femora_solver.compile.compiler import Compiler
from femora_solver.analysis.runner import Runner
from femora_solver.state.state import State, FieldState
from femora_solver.loads.load_plan import normalize_time_fn
from typing import Optional, Any, Union, List, Dict
import numpy as np
import jax


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
        self._profiler = None
        
    def add_nodes(self, coords: Union[List[List[float]], np.ndarray, jax.Array]):
        self._nodes = jnp.array(coords)
        self._num_nodes = len(coords)
        self._dirty = 4
        
    def add_elements(
        self,
        conn: Union[List[List[int]], np.ndarray, jax.Array],
        family: str = "Hex8",
        material: Optional[str] = None,
        block_id: Optional[str] = None,
        section_params: Any = None
    ):
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
        
    def add_material(self, name: str, kind: str, **kwargs):
        if name in self._materials:
            raise ValueError(f"Material '{name}' already exists in this model.")
        self._materials[name] = {"kind": kind, **kwargs}
        self._dirty = 4
        
    def add_constraint(
        self,
        name: str,
        kind: str,
        node_indices: Union[List[int], np.ndarray, jax.Array],
        dofs: Any = None,
        components: Optional[List[int]] = None,
        field: str = "U"
    ):
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
        
    def add_load(
        self,
        name: str,
        kind: str,
        field: str,
        node_indices: Union[List[int], np.ndarray, jax.Array],
        force: Union[List[float], np.ndarray, jax.Array],
        time_fn: Any = None
    ):
        # Validation: Ensure force matches node count (allow single vector for broadcasting)
        f_arr = jnp.array(force)
        n_nodes = len(node_indices)
        if f_arr.ndim == 2:
            n_forces = f_arr.shape[0]
            if n_forces != 1 and n_forces != n_nodes:
                raise ValueError(
                    f"Load '{name}' mismatch: {n_nodes} nodes but {n_forces} force vectors provided. "
                    "Must provide exactly one vector (to be broadcast) or one vector per node."
                )

        # Validation: Check and normalize the time function
        time_fn_spec = normalize_time_fn(time_fn)

        self._loads.append({
            "name": name,
            "kind": kind,
            "field": field,
            "node_indices": node_indices,
            "force": force,
            "time_fn": time_fn_spec,
        })
        self._dirty = max(self._dirty, 1)
        
    def add_recorder(
        self,
        name: str,
        kind: str,
        node_indices: Union[List[int], np.ndarray, jax.Array],
        field: str,
        interval: int,
        file: str
    ):
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

    def run(
        self,
        dt,
        time,
        chunk_size: Optional[int] = None,
        progress: bool = False,
        progress_every: int = 1,
        sync_progress: Optional[bool] = None,
        profile: bool = False,
        profile_export: Optional[str] = None,
    ):
        """Run the simulation.

        Parameters
        ----------
        dt : float
            Time step size.
        time : float
            Total simulation time.
        chunk_size : int, optional
            Number of steps per lax.scan chunk.
        progress : bool
            Print progress to stdout.
        progress_every : int
            Print every N chunks.
        sync_progress : bool, optional
            Force GPU sync per chunk for accurate progress.
        profile : bool
            Enable phase-level profiling (uses decomposed stepping).
        profile_export : str, optional
            Path to export profiling results as JSON.
        """
        profiler = None
        if profile:
            from femora_solver.profiling.timer import SolverProfiler
            profiler = SolverProfiler()
            profiler.mark_start()

        if self._dirty >= 4:
            self._execution_plan = self._compiler.full_compile(self, profiler=profiler)
            if self._state is None:
                self._init_state()
        self._dirty = 0
        
        self._state = self._runner.run(
            self._execution_plan,
            self._state,
            dt,
            time,
            chunk_size=chunk_size,
            progress=progress,
            progress_every=progress_every,
            sync_progress=sync_progress,
            profiler=profiler,
        )

        if profiler is not None:
            profiler.mark_end()
            self._profiler = profiler
            print(profiler.summary())
            if profile_export:
                profiler.to_json(profile_export)
                print(f"Profile exported to: {profile_export}")
