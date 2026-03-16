# Femora Solver — Full Implementation Design Specification (v2)

---

## 0. How to use this document

This document is a complete, implementation-ready handoff specification for building
**Femora Solver**. It extends the original design with the following additions:

- A phased implementation plan with explicit milestones
- A corrected `lax.scan` / stepping model that handles staged analysis properly
- A fully specified `CommPlan` with JAX-compatible data structures for ghost exchange
- A hardened recorder async design with buffer management and backpressure handling
- Class skeletons with method signatures for the first implementation pass
- Explicit rules for what NOT to do

Read sections 1–4 for architecture context. Section 5 is the implementation plan.
Sections 6–18 are the full detailed spec. Section 19 contains class skeletons.

---

## 1. Project identity

**Project name:** `Femora Solver`

**Positioning:** a separate project that serves as the solver engine for Femora.
If successful, it can later be merged back into the main Femora ecosystem.

**Primary target:** a GPU-first, JAX-optimized, multi-node, multi-GPU explicit FEM solver
for very large structural dynamics problems.

**First physics target:**
- explicit transient structural dynamics
- lumped mass
- matrix-free internal force evaluation
- mixed element families
- arbitrary mixed DOF signatures
- staged analysis workflow

**Backend requirement:** the design must be explicitly optimized for modern JAX:
- `jax.Array`
- explicit sharding
- multi-process / multi-host initialization
- distributed arrays
- `lax.scan` for stepping (in chunk mode — see Section 10)
- async host copies for recorder output
- no design that depends on old single-device-only assumptions

---

## 2. One-sentence summary

**Femora Solver is a separate JAX-optimized, GPU-resident, field-based, block-executed,
multi-node explicit FEM solver project with a Model-centric API, designed for
million-element simulations, arbitrary mixed DOF signatures, interfaces/MPC constraints,
and staged analysis workflows.**

---

## 3. Core design goals

Femora Solver must be designed for:

- millions of elements
- mixed element families
- arbitrary field signatures, not fixed `ndf`
- GPU residency during analysis
- multi-node / multi-GPU scaling from the start
- minimal CPU↔GPU transfer
- stage-based workflows
- interfaces and MPC/equalDOF constraints
- JAX-friendly data layout and execution

The user-facing API should feel simple and model-centric, similar in spirit to OpenSees:

```python
model = Model("bridge")

model.add_nodes(...)
model.add_elements(...)
model.add_material(...)
model.add_constraint(...)
model.add_load(...)
model.add_recorder(...)

model.run(dt=1e-5, time=5.0)

model.remove_load("quake")
model.add_recorder(...)
model.run(dt=1e-5, time=2.0)
```

The second `run(...)` must continue from the current state.

---

## 4. High-level architecture: three layers

The architecture is split into three layers.

### A. Model layer

This is the public engineering model. The user edits this.

It owns:
- nodes
- elements
- materials
- sections
- sets
- constraints
- loads
- recorders
- stage definitions
- current state handle
- internal compiled execution plan
- dirty flags

### B. Execution layer — ExecutionPlan

This is the compiled, GPU/distributed representation of the model.

It exists because the editable engineering model is not suitable for efficient JAX
execution at scale.

It owns:
- field-space layouts
- field-local indexing
- partition metadata
- owned/ghost maps
- element blocks
- coupling/interface blocks
- constraint plans
- load plans
- recorder plans
- mass arrays
- history-variable layouts
- sharding/device layout
- CommPlan (see Section 14)

### C. State layer — State

This is the evolving numerical solution. It stays on GPU during analysis.

It owns:
- current time
- current step
- field values (per field, per partition)
- velocities (per field)
- accelerations (per field)
- material history variables
- optional energy counters
- optional recorder staging buffers

The `Model` owns the current `State` handle, but `State` is logically separate
from the engineering description.

---

## 5. Phased implementation plan (CRITICAL — read first)

> **Warning:** building all 17 sections simultaneously before a single kernel runs
> is the primary risk. The module tree is comprehensive, but implementation must be
> staged. Do not proceed to later phases without validating earlier phases first.

### Phase 1 — Vertical slice (single GPU, single field, single element type)

**Goal:** explicit dynamics working end-to-end on GPU.

**Scope:**
- single partition, single device
- single field space: `U(3)` only
- single element family: Hex8
- single material: linear elastic
- lumped mass
- simple Dirichlet BCs (fixed DOFs)
- nodal load with a time function
- simple nodal recorder (async CPU copy)
- no interfaces, no MPC, no equalDOF

**Validation target:**
- wave propagation in a unit cube, 1M Hex8 elements
- verify against analytical solution (P-wave arrival time)
- measure GPU utilization and throughput

**Do not move to Phase 2 until Phase 1 passes validation.**

### Phase 2 — Mixed fields and constraints

**Scope:**
- add `R(3)` field space
- add Beam2 element with `U, R` signature
- add equalDOF constraint system
- add general linear MPC
- verify beam-on-solid setup

### Phase 3 — Interfaces and couplings

**Scope:**
- beam-solid interface
- solid-solid tie
- embedded element coupling
- verify energy conservation across interface

### Phase 4 — Multi-GPU, single node

**Scope:**
- 2–4 GPU single node
- domain partitioning
- CommPlan ghost exchange
- verify results match single-GPU reference

### Phase 5 — Multi-node distributed

**Scope:**
- multi-node MPI-style via `jax.distributed`
- full CommPlan exchange
- distributed recorder output (per-rank files)

### Phase 6 — Additional element families

**Scope:**
- Tet4, Prism6
- custom element plugin interface
- mixed element runs

---

## 6. The most important abstraction: FieldSpace

The solver must **not** be designed around a fixed universal node `ndf`.

That would break as soon as you introduce:
- `R3 + S6`
- custom 5-component unknowns
- mixed beam/solid/interface fields
- poromechanics or other coupled formulations later

### FieldSpace

A field space is a named unknown with arbitrary component count.

```python
FieldSpace("U", 3)   # displacement
FieldSpace("R", 3)   # rotation
FieldSpace("P", 1)   # pressure
FieldSpace("S", 6)   # custom
FieldSpace("Q", 5)   # custom
```

### Key rule

Nodes do not fundamentally "have one ndf."
Nodes **participate in one or more field spaces**.

Examples:
- a solid node belongs to `U`
- a beam node belongs to `U` and `R`
- a special element node belongs to `R` and `S`

### Internal state storage

Each field is stored separately:

```python
state.fields["U"].u   # (nU_nodes, 3) — displacement
state.fields["U"].v   # (nU_nodes, 3) — velocity
state.fields["U"].a   # (nU_nodes, 3) — acceleration
state.fields["U"].m   # (nU_nodes, 3) — lumped mass (per component)

state.fields["R"].u   # (nR_nodes, 3)
state.fields["R"].v   # (nR_nodes, 3)
state.fields["R"].a   # (nR_nodes, 3)
state.fields["R"].m   # (nR_nodes, 3)

state.fields["S"].u   # (nS_nodes, 6)
# ... etc
```

This is the correct architecture for arbitrary mixed DOFs at scale.

---

## 7. Element architecture

Elements must be grouped into **ElementBlocks**, not executed as individual Python objects.

### ElementBlock

An ElementBlock is a group of elements sharing the same execution signature.

Execution signature includes:
- element family (Hex8, Tet4, Beam2, Prism6, custom...)
- interpolation order
- field signature (e.g. `["U"]`, `["U", "R"]`, `["R", "S"]`, `["Q"]`)
- quadrature rule
- material kernel type
- section type if needed
- history-variable layout

Examples:
- `Hex8 + ["U"] + linear_elastic`
- `Hex8 + ["U"] + J2_plasticity`
- `Beam2 + ["U", "R"] + elastic_section`
- `Custom + ["R", "S"]`

### Why blocks matter

This avoids branch-heavy mega-kernels and fits JAX/GPU execution much better.
Python dispatch per element is not acceptable in the hot path.

### Element block data layout

Each ElementBlock stores its connectivity as a dense integer array:

```python
block.connectivity   # (n_elements, n_nodes_per_elem) — field-local node indices
block.material_ids   # (n_elements,) — index into material parameter table
block.history        # (n_elements, n_qp, n_hist_vars) — history variables
```

Field-local node indices are indices into `state.fields[fname].u`, not global node IDs.

### Element responsibilities

Element kernels compute:
- internal force contributions (scatter to field arrays)
- stable time-step estimate (critical dt per element)
- material/history updates
- no stiffness matrix — fully matrix-free

---

## 8. Constraints and interfaces architecture

Constraints and interfaces must NOT be buried inside element-family logic.
They need a separate compiled layer.

### Constraint system

Handles:
- single-point constraints (fixed DOFs)
- `equalDOF`
- general linear MPCs

All compile into **field relations** — linear equations over field DOFs.

`equalDOF(node_i, node_j, Ux)` becomes:

```
U_i^x - U_j^x = 0
```

A general MPC is:

```
sum_k c_k * q_k = rhs
```

The constraint plan stores these as sparse coefficient arrays and applies them
as either:
- projection (set DOF values directly) for Dirichlet BCs
- penalty or Lagrange multiplier for general MPCs

Implementation approach: compile all constraints into a `ConstraintPlan` containing:

```python
plan.dirichlet_field     # str: field name
plan.dirichlet_indices   # (n_fixed,) int array — field-local DOF flat indices
plan.dirichlet_values    # (n_fixed,) or callable(t) — values

plan.mpc_lhs_rows        # sparse representation
plan.mpc_lhs_cols
plan.mpc_lhs_coeffs
plan.mpc_rhs             # (n_mpc,)
```

### Coupling/interface system

Handles:
- beam-solid interfaces
- solid-solid ties
- embedded couplings
- interpolation-based ties

Each interface is built from:
- source entity set (nodes or elements)
- target entity set (nodes or elements)
- field/components on each side
- interpolation/tying rule
- enforcement method (penalty, Lagrange, direct)

The interface system works from field access and topology/geometry, not from
"one uniform element family block." A beam can tie to a solid region even if
the solid region spans multiple bulk blocks with different materials.

---

## 9. GPU-resident execution philosophy

1. CPU reads mesh and high-level model data
2. CPU compiles or updates the ExecutionPlan
3. Arrays are placed on devices with explicit layout/sharding
4. Analysis state stays on GPU
5. Stepping, internal forces, local updates stay on GPU
6. Data is copied to CPU only when explicitly requested (recorders, checkpoints)

### Keep on GPU

- field states (u, v, a, m per field)
- internal force work arrays
- element connectivity arrays
- history variables
- partition-local metadata
- coupling/constraint working arrays
- recorder staging buffers

### Keep on CPU

- model editing
- mesh reading
- setup and orchestration
- output file writing
- checkpoint coordination

---

## 10. Time stepping design — CRITICAL: lax.scan in chunk mode

> **This section corrects a key design risk from naive `lax.scan` use.**

### The problem with naive lax.scan

`lax.scan` requires:
- a pure function loop body
- fixed array shapes throughout the scan
- no Python-level decisions during stepping

This conflicts with:
- staged analysis (loads/recorders can change between runs)
- adaptive time stepping (variable dt)
- recorder sampling at irregular intervals
- constraint changes mid-stage

**You cannot run the entire analysis inside a single `lax.scan` call.**

### Correct design: chunk-scan hybrid

Use a two-level loop:

**Outer loop (Python):** orchestrates stages and recorder output
**Inner loop (lax.scan):** runs a fixed chunk of N steps compiled

```python
def run(dt, time):
    total_steps = int(time / dt)
    chunk_size  = plan.recorder_interval  # or a fixed chunk like 100 or 1000

    for chunk_start in range(0, total_steps, chunk_size):
        n_steps = min(chunk_size, total_steps - chunk_start)

        # lax.scan over a fixed chunk — fully compiled
        state, recorder_buffer = lax.scan(
            step_fn,            # pure: (state, t) -> (state, output)
            state,
            jnp.arange(chunk_start, chunk_start + n_steps) * dt
        )

        # async copy recorder buffer to CPU
        _enqueue_recorder_write(recorder_buffer)

        # Python can check for stage changes here — zero GPU cost
        if _stage_dirty():
            plan = _recompile_partial(plan)
```

### step_fn contract

```python
def step_fn(state: State, t: float) -> tuple[State, RecorderPayload]:
    # 1. ghost exchange
    # 2. compute internal forces block by block
    # 3. compute coupling/interface contributions
    # 4. apply loads
    # 5. apply constraints
    # 6. update acceleration
    # 7. update velocity
    # 8. update displacement/fields
    # 9. sample recorder payload
    return new_state, recorder_payload
```

`step_fn` must be a pure function. All configuration (load amplitudes, constraint
masks, recorder indices) must be compiled into the arrays it references, not
injected via Python closures that change at runtime.

### Chunk size selection

| Scenario | Recommended chunk size |
|---|---|
| Development / debugging | 1 (effectively no scan) |
| Single recorder interval | recorder_interval steps |
| No recorder | 100–1000 steps |
| Maximum throughput | as large as GPU memory allows |

### Adaptive dt

If adaptive dt is needed in the future, run the critical-dt estimate once per chunk
in Python, then pass a fixed dt into that chunk's `lax.scan`. Do not attempt to
vary dt inside a scan.

---

## 11. Recorder architecture

### Recorder types

- full field recorder (all DOFs of a field)
- node-set recorder (subset of nodes)
- element-set recorder (strains, stresses at selected elements)
- reduced/statistics recorder (norms, energies)
- checkpoint recorder

### Correct async design

The recorder must not block the stepping loop. Design:

1. Inside `step_fn`: extract a small payload (selected node indices/values) and
   return it as the scan output — this stays on GPU.
2. After each chunk's `lax.scan`: call `jax.device_get(recorder_buffer)`
   **asynchronously** — this enqueues a device-to-host copy and returns
   immediately. JAX dispatches this asynchronously.
3. A background Python thread (or `concurrent.futures.ThreadPoolExecutor`) waits
   on the copy future and writes to HDF5 or VTK-HDF.

### Backpressure management

If the writer thread falls behind the stepper, you must not drop data.
Implementation rule: use a bounded queue of fixed size (e.g. 8 chunks).
If the queue is full, the stepping loop blocks. This is acceptable because
at that point the GPU is running at recorder-write speed, not solver speed.

```python
RECORDER_QUEUE_DEPTH = 8   # tune based on GPU speed vs. disk speed

recorder_queue = queue.Queue(maxsize=RECORDER_QUEUE_DEPTH)

# writer thread
def _writer_thread():
    while True:
        item = recorder_queue.get()
        if item is None:
            break
        _write_to_hdf5(item)

# stepper
for chunk in chunks:
    state, buf = lax.scan(step_fn, state, ts)
    # this blocks if queue is full — intentional backpressure
    recorder_queue.put(jax.device_get(buf))
```

### Recorder plan compile-time data

The `RecorderPlan` must contain only static integer index arrays. It must not
contain Python callables or dynamic references.

```python
plan.recorder_field_name      # str
plan.recorder_node_indices    # (n_rec_nodes,) int array — field-local
plan.recorder_interval        # int — steps per sample
plan.recorder_component_mask  # (ndf,) bool — which components
```

### Distributed recorder rule

For distributed runs, each rank writes its own recorder file. Global gather is
**not** performed during analysis. Files are merged in post-processing if needed.

---

## 12. CommPlan — fully specified ghost exchange

> **This section addresses the most underspecified part of the original design.**

### Why CommPlan must be JAX-compatible data structures

For JAX distributed execution, communication cannot rely on Python dicts of sets.
The CommPlan must be expressed entirely as integer index arrays so that
gather/scatter operations can be expressed as JAX `jnp.take` / `jnp.scatter_add`
operations on device arrays.

### CommPlan structure

```python
@dataclass
class FieldCommPlan:
    field_name: str
    ncomp: int

    # Export side (what this rank sends to neighbors)
    export_rank:    np.ndarray   # (n_exports,) int — destination rank
    export_indices: np.ndarray   # (n_exports,) int — field-local owned node index

    # Import side (what this rank receives from neighbors)
    import_rank:    np.ndarray   # (n_imports,) int — source rank
    import_indices: np.ndarray   # (n_imports,) int — field-local ghost node index

    # For JAX collective ops: permutation arrays
    pack_perm:    jax.Array      # (n_exports,) — indices to gather before send
    unpack_perm:  jax.Array      # (n_imports,) — indices to scatter after receive

@dataclass
class CommPlan:
    n_ranks:   int
    this_rank: int
    fields:    dict[str, FieldCommPlan]
    # Interface exchange metadata if needed
    interface_comm: list[InterfaceCommPlan]
```

### Ghost exchange algorithm (JAX-compatible)

```python
def exchange_ghosts(state: State, comm_plan: CommPlan) -> State:
    new_fields = {}
    for fname, fcomm in comm_plan.fields.items():
        u = state.fields[fname].u    # (n_owned + n_ghost, ncomp)

        # pack: gather export values
        export_vals = u[fcomm.pack_perm]     # (n_exports, ncomp)

        # communicate (this is the only non-JAX step — use jax.experimental.multihost_utils
        # or a thin MPI wrapper that exchanges export_vals → import_vals)
        import_vals = _comm_exchange(export_vals, fcomm, comm_plan)

        # unpack: scatter import values into ghost region
        u = u.at[fcomm.unpack_perm].set(import_vals)

        new_fields[fname] = replace(state.fields[fname], u=u)

    return replace(state, fields=new_fields)
```

### Key rule

Ghost exchange must happen **once per step**, before internal force computation.
It must not happen inside element kernels.

### Single-device case

For Phase 1 (single device), CommPlan is empty. `exchange_ghosts` is a no-op.
This must be designed in from the start so that the stepping loop does not need
to be rewritten when moving to multi-GPU.

---

## 13. Staged analysis — dirty flag rebuild system

Stage-based analysis is a first-class requirement.

The user must be able to:
- run a stage
- remove/add loads
- add/remove recorders
- adjust some constraints
- continue from the current state

### Dirty rebuild levels

The Model tracks dirty levels. When `model.run(...)` is called, the rebuild
level determines what must be recompiled.

**Level 0 — no rebuild.** Just continue stepping with the current plan.

**Level 1 — lightweight update.** Loads or recorders changed. Recompile
`LoadPlan` and `RecorderPlan` only. State is untouched. Cost: milliseconds.

**Level 2 — constraint/coupling patch.** Constraint masks or coupling
definitions changed without topology changes. Recompile `ConstraintPlan`
and `CouplingPlan`. State is untouched. Cost: low.

**Level 3 — block rebuild.** Element grouping, field membership maps, or
material-block layouts changed. Rebuild `ElementBlock` list and re-JIT the
step function. State must be re-mapped if field sizes changed. Cost: seconds.

**Level 4 — full rebuild.** Topology, field definitions, or partitioning
changed. Full recompile including CommPlan, sharding, all block layouts.
Cost: may be tens of seconds for large models.

### Dirty flag propagation

```python
# in Model
def add_load(self, load):
    self._loads.append(load)
    self._dirty = max(self._dirty, DirtyLevel.LOAD)

def add_element(self, elem):
    self._elements.append(elem)
    self._dirty = max(self._dirty, DirtyLevel.FULL)

def run(self, dt, time):
    if self._dirty >= DirtyLevel.FULL:
        self._execution_plan = self._compiler.full_compile(self)
    elif self._dirty >= DirtyLevel.BLOCK:
        self._execution_plan = self._compiler.rebuild_blocks(self._execution_plan, self)
    elif self._dirty >= DirtyLevel.CONSTRAINT:
        self._execution_plan = self._compiler.patch_constraints(self._execution_plan, self)
    elif self._dirty >= DirtyLevel.LOAD:
        self._execution_plan = self._compiler.update_load_plan(self._execution_plan, self)
    self._dirty = DirtyLevel.NONE
    self._runner.run(self._execution_plan, self._state, dt, time)
```

---

## 14. JAX-specific design rules

### Use these JAX concepts as first-class design assumptions

- `jax.Array` as the main array model — never `np.ndarray` for GPU-resident data
- explicit sharding with `jax.sharding.NamedSharding` or `PositionalSharding`
- multi-process execution via `jax.distributed.initialize()`
- `lax.scan` for inner loop over fixed step chunks (see Section 10)
- `vmap` for batching over elements within a block where applicable
- `shard_map` for partition-local kernels when needed
- `jax.device_put` with explicit sharding for array placement
- `jax.device_get` for async host copies (recorder output)

### Do NOT design around

- Python loops in the hot path (element-by-element dispatch)
- Python object dispatch during stepping
- monolithic ragged state storage (different fields mixed in one array)
- implicit host mirroring
- single-device-only assumptions that require refactoring for multi-GPU
- `np.ndarray` on CPU as the primary array type for solver state

### Multi-node / multi-GPU initialization

```python
# At process startup, before any JAX operations
jax.distributed.initialize()   # reads COORDINATOR_ADDRESS, NUM_PROCESSES, PROCESS_ID from env

# Confirm device layout
devices = jax.devices()
mesh = jax.sharding.Mesh(np.array(devices).reshape(n_x, n_y), ("x", "y"))
```

---

## 15. Distributed architecture

### Partitioning

Partition by subdomain. Use METIS or a simple recursive bisection for Phase 4+.

Each partition owns:
- owned nodes per field (contiguous block of field-local indices: 0..n_owned-1)
- ghost nodes per field (appended block: n_owned..n_owned+n_ghost-1)
- local element blocks
- local material state
- local lumped mass
- local interface/coupling data
- local recorder buffers

### Ownership convention

Owned nodes are always stored first in field arrays. Ghost nodes follow.
This convention must be enforced by the compiler and never violated.

```
field array layout:  [owned_0, ..., owned_{n-1}, ghost_0, ..., ghost_{m-1}]
                      ^--- n_owned --->  ^--- n_ghost --->
```

Internal force scatter: scatter only to owned nodes. Ghost contributions are
discarded (they are owned by neighbor ranks and will be computed there).

### Principle

Do not design around a monolithic global vector if you want scalability with
arbitrary field signatures.

---

## 16. Recommended project structure

```
femora-solver/
  pyproject.toml
  README.md
  PHASES.md         # implementation phase checklist

  src/
    femora_solver/
      model/
        model.py              # public Model class
        stage.py              # stage definition and dirty tracking
        sets.py               # NodeSet, ElementSet
        commands.py           # add_nodes, add_elements, etc. mixins

      mesh/
        nodes.py              # node coordinate storage
        topology.py           # element-node connectivity (raw, pre-compile)
        partition.py          # partitioning (METIS wrapper or simple bisection)
        surfaces.py           # surface extraction for interface detection

      fields/
        field_space.py        # FieldSpace definition
        field_manager.py      # manages all active field spaces in a model
        indexing.py           # node-to-field-local index maps

      elements/
        families/
          hex8.py
          tet4.py
          beam2.py
          prism6.py
          custom.py           # plugin interface
        kernels/
          solid_u.py          # JAX kernel: Hex8/Tet4/Prism6 + U field
          beam_ur.py          # JAX kernel: Beam2 + U,R fields
          custom_rs.py
          custom_q.py
        block.py              # ElementBlock dataclass + dispatch

      materials/
        base.py               # material kernel protocol
        linear_elastic.py
        j2.py
        custom.py

      sections/
        beam_section.py
        custom.py

      constraints/
        spc.py                # single-point constraint definition
        equal_dof.py          # equalDOF definition
        mpc.py                # general MPC definition
        constraint_manager.py # collects all constraints for a model
        constraint_plan.py    # compiled ConstraintPlan (index arrays)

      couplings/
        interface.py          # interface definition (source, target, rule)
        beam_solid.py         # beam-solid tying kernel
        solid_solid.py        # solid-solid tie kernel
        embedded.py           # embedded element coupling kernel
        interface_manager.py
        coupling_block.py     # CouplingBlock dataclass
        coupling_plan.py      # compiled CouplingPlan

      loads/
        nodal.py              # nodal force definition
        body.py               # body force definition
        surface.py            # surface traction definition
        time_function.py      # load time functions
        load_plan.py          # compiled LoadPlan (index + amplitude arrays)

      compile/
        compiler.py           # main Compiler class — all rebuild levels
        execution_plan.py     # ExecutionPlan dataclass
        block_builder.py      # builds ElementBlock list from raw model
        field_layout_builder.py # builds field-local indexing from mesh
        recorder_plan.py      # compiled RecorderPlan
        comm_plan.py          # builds CommPlan from partition metadata

      state/
        state.py              # State dataclass (field arrays, history)
        field_state.py        # per-field u/v/a/m arrays
        history.py            # material history variable layout
        checkpoint.py         # save/load checkpoint

      analysis/
        runner.py             # outer Python loop (chunk orchestration)
        transient_explicit.py # step_fn pure function (inner lax.scan body)
        stepper.py            # lax.scan wrapper + chunk management
        stable_dt.py          # critical dt estimate

      recorder/
        recorder_plan.py      # (also referenced from compile/)
        field_recorder.py     # extract field payload from state
        set_recorder.py       # extract node/element set payload
        checkpoint_recorder.py
        async_writer.py       # background thread + bounded queue
        writer_hdf5.py        # HDF5 output
        writer_vtkhdf.py      # VTK-HDF output

      distributed/
        init.py               # jax.distributed.initialize() wrapper
        sharding.py           # device mesh and sharding specs
        exchange.py           # ghost exchange implementation
        reductions.py         # global reductions (energy norms, etc.)

      backend/
        jax_utils.py          # JAX helpers, device placement
        device_arrays.py      # utilities for placing arrays on devices
        kernels.py            # shared JAX kernel utilities

  tests/
    phase1/
      test_wave_propagation.py    # Phase 1 validation
    phase2/
      test_beam_solid.py
    phase3/
      test_interface_energy.py
    phase4/
      test_multigpu_consistency.py
```

---

## 17. Public API

```python
model = Model("example")

# Build model
model.add_nodes(coords)                      # coords: (N, 3) array
model.add_elements(conn, family="Hex8",
                   material="steel")
model.add_material("steel", kind="LinearElastic",
                   E=200e9, nu=0.3, rho=7800.0)
model.add_section("beam_sec", kind="Rectangular", b=0.3, h=0.5)
model.add_constraint("fixed_base", kind="SPC",
                     node_set="base", dofs=["Ux","Uy","Uz"])
model.add_load("gravity", kind="Body",
               field="U", components=[0, 0, -9.81],
               time_fn=lambda t: 1.0)
model.add_recorder("disp_rec", kind="NodeSet",
                   node_set="output_nodes",
                   field="U", interval=100,
                   file="disp.h5")

# Run stage 1
model.run(dt=1e-5, time=5.0)

# Modify and continue
model.remove_load("gravity")
model.add_load("quake", kind="Nodal",
               node_set="base", field="U",
               time_series=accel_table)
model.run(dt=1e-5, time=30.0)

# Checkpoint
model.save_checkpoint("chkpt_01")
model.reset_state()
model.load_checkpoint("chkpt_01")
model.run_steps(dt=1e-5, steps=1000)
```

---

## 18. Internal class skeletons with method signatures

These are the primary classes to implement in Phase 1. Later phases extend these.

### FieldSpace

```python
@dataclass(frozen=True)
class FieldSpace:
    name: str      # e.g. "U"
    ncomp: int     # e.g. 3

    def __repr__(self):
        return f"FieldSpace({self.name!r}, {self.ncomp})"
```

### FieldState

```python
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
```

### State

```python
@dataclass
class State:
    time: float
    step: int
    fields: dict[str, FieldState]        # keyed by field name
    history: dict[str, jax.Array]        # keyed by block id
    energy_kinetic: float | None
    energy_internal: float | None

    def replace(self, **kwargs) -> "State":
        return dataclasses.replace(self, **kwargs)
```

### ElementBlock

```python
@dataclass
class ElementBlock:
    block_id: str
    family: str                          # "Hex8", "Beam2", etc.
    field_names: list[str]               # e.g. ["U"] or ["U", "R"]
    connectivity: jax.Array              # (n_elem, n_nodes_per_elem) field-local
    material_params: jax.Array           # (n_elem, n_params) or (n_params,) if uniform
    history_shape: tuple[int, ...]       # (n_elem, n_qp, n_hist_vars)
    quadrature_rule: QuadratureRule

    def compute_internal_forces(
        self,
        field_u: dict[str, jax.Array],  # field_name -> (n_nodes, ncomp)
        history: jax.Array,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        # returns (force_contributions per field, updated_history)
        ...

    def compute_stable_dt(
        self,
        field_u: dict[str, jax.Array],
        material_params: jax.Array,
    ) -> float:
        ...
```

### ExecutionPlan

```python
@dataclass
class ExecutionPlan:
    fields: dict[str, FieldSpace]
    element_blocks: list[ElementBlock]
    constraint_plan: ConstraintPlan
    coupling_plan: CouplingPlan
    load_plan: LoadPlan
    recorder_plan: RecorderPlan
    comm_plan: CommPlan
    mass_arrays: dict[str, jax.Array]    # field_name -> (n_owned, ncomp)
    sharding: dict[str, jax.sharding.Sharding]
    chunk_size: int
```

### ConstraintPlan

```python
@dataclass
class ConstraintPlan:
    # Dirichlet BCs per field
    dirichlet: dict[str, DirichletData]  # field_name -> DirichletData

    # MPC (general linear)
    mpc_rows: jax.Array     # (n_mpc_terms,) — which constraint
    mpc_cols: jax.Array     # (n_mpc_terms,) — which flat DOF index
    mpc_coeffs: jax.Array   # (n_mpc_terms,) — coefficient
    mpc_rhs: jax.Array      # (n_mpc,)

@dataclass
class DirichletData:
    flat_indices: jax.Array   # (n_fixed,) — flat index into field u array
    values: jax.Array         # (n_fixed,) — prescribed value (static)
    # time-varying values handled via LoadPlan amplitude scaling
```

### LoadPlan

```python
@dataclass
class LoadPlan:
    nodal_loads: list[NodalLoadData]
    body_loads: list[BodyLoadData]
    surface_loads: list[SurfaceLoadData]

@dataclass
class NodalLoadData:
    field_name: str
    node_indices: jax.Array   # (n_loaded,) field-local
    force_pattern: jax.Array  # (n_loaded, ncomp)
    time_fn_id: int           # index into time function table
```

### RecorderPlan

```python
@dataclass
class RecorderPlan:
    recorders: list[RecorderEntry]
    chunk_size: int

@dataclass
class RecorderEntry:
    recorder_id: str
    field_name: str
    node_indices: jax.Array     # (n_rec,) field-local
    component_mask: jax.Array   # (ncomp,) bool
    interval: int               # every N steps
    output_path: str
```

### CommPlan

```python
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
    # empty in Phase 1 (single device)
```

### Compiler

```python
class Compiler:
    def full_compile(self, model: "Model") -> ExecutionPlan:
        """Level 4 rebuild: topology + fields + partitioning + all plans."""
        ...

    def rebuild_blocks(self, plan: ExecutionPlan, model: "Model") -> ExecutionPlan:
        """Level 3 rebuild: element grouping, block layouts, re-JIT step_fn."""
        ...

    def patch_constraints(self, plan: ExecutionPlan, model: "Model") -> ExecutionPlan:
        """Level 2 rebuild: constraint and coupling plans only."""
        ...

    def update_load_plan(self, plan: ExecutionPlan, model: "Model") -> ExecutionPlan:
        """Level 1 rebuild: load and recorder plans only."""
        ...
```

### Runner

```python
class Runner:
    def run(
        self,
        plan: ExecutionPlan,
        state: State,
        dt: float,
        time: float,
    ) -> State:
        """Outer Python loop over chunks. Returns final state."""
        total_steps = int(round(time / dt))
        chunk_size  = plan.chunk_size
        step_fn     = self._build_step_fn(plan)  # jit-compiled

        for start in range(0, total_steps, chunk_size):
            n = min(chunk_size, total_steps - start)
            ts = jnp.arange(start, start + n, dtype=jnp.float32) * dt
            state, rec_buf = lax.scan(step_fn, state, ts)
            self._enqueue_recorder(rec_buf)

        self._flush_recorder()
        return state

    def _build_step_fn(self, plan: ExecutionPlan):
        """Build and jit-compile the step function for this plan."""
        ...
```

---

## 19. What NOT to do — explicit anti-patterns

These patterns will be encountered during implementation. Avoid them.

| Anti-pattern | Correct approach |
|---|---|
| Global node `ndf` field on every node | FieldSpace with per-field node lists |
| Python loop over elements in step_fn | ElementBlock with vmap/scan over block |
| Single `lax.scan` over entire analysis time | Chunk-scan hybrid (Section 10) |
| `numpy.ndarray` for GPU-resident state | `jax.Array` placed on device |
| MPI barrier every step | Ghost exchange once per step, no global barriers |
| Global state vector (all fields concatenated) | Separate per-field arrays |
| Hardcoded equalDOF, MPC, SPC as separate code paths | All compile to linear field relations |
| Recorder writes inside step_fn | Collect payload, async write after chunk |
| Python dict/set in CommPlan for ghost maps | Integer index arrays (pack_perm, unpack_perm) |
| Monolithic mega-kernel that branches on element type | One kernel per ElementBlock family |
| Variable-length arrays in lax.scan body | Fixed shapes; pad if needed |
| Calling `jax.device_get` inside the hot path | Only after lax.scan, outside step_fn |
| Building full stiffness matrix | Matrix-free; compute internal forces directly |

---

## 20. Validation benchmarks

These must be run at each phase milestone.

### Phase 1 — wave propagation

- 1M Hex8 elements in a 100×100×100 m cube
- P-wave speed: `c_p = sqrt((lambda + 2*mu) / rho)`
- Apply impulse load at center of top face at t=0
- Verify arrival time at bottom face within 1% of analytical
- Measure: wall time per 1000 steps, GPU utilization, memory footprint

### Phase 2 — beam-solid cantilever

- Cantilever beam modeled with solid elements (Hex8)
- Beam tip load applied as nodal force
- Verify tip deflection against Euler-Bernoulli
- Add beam2 elements alongside and verify DOF coupling via equalDOF

### Phase 3 — interface energy conservation

- Two solid blocks tied via solid-solid interface
- Free vibration from initial displacement
- Verify total mechanical energy conservation to within 1% over 1000 steps

### Phase 4 — multi-GPU consistency

- Same Phase 1 problem, split across 2 then 4 GPUs
- Results must match single-GPU to within floating-point tolerance (1e-5 relative)

---

## 21. Dependencies

**Core:**
- `jax[cuda]` — primary compute backend
- `numpy` — CPU-side mesh/setup operations
- `h5py` — HDF5 recorder output

**Mesh/partitioning:**
- `pymetis` — graph partitioning (Phase 4+)

**Testing:**
- `pytest`
- `numpy.testing`

**Optional:**
- `vtkhdf` or `pyvista` — VTK-HDF output
- `mpi4py` — only if thin MPI wrapper needed for ghost exchange in multi-node

**Python version:** 3.11+

**JAX version:** JAX 0.4.30+ (required for current distributed API)

---

*End of specification.*
