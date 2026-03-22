import queue
import threading
import jax
import h5py
import numpy as np
import time as _time
from typing import Optional
from femora_solver.state.state import State
from femora_solver.compile.execution_plan import ExecutionPlan
from femora_solver.analysis.transient_explicit import build_step_fn, build_profiled_step_phases
import jax.numpy as jnp

class AsyncWriter:
    def __init__(self, plan: ExecutionPlan, profiler=None):
        self.plan = plan
        self.profiler = profiler
        self.queue = queue.Queue(maxsize=8)
        self.thread = threading.Thread(target=self._writer_thread, daemon=True)
        self.thread.start()
        
    def enqueue(self, payload):
        self.queue.put(payload)
        
    def close(self):
        self.queue.put(None)
        self.thread.join()
        
    def _writer_thread(self):
        # Open files for recorders
        files = {}
        for rec in self.plan.recorder_plan.recorders:
            files[rec.recorder_id] = h5py.File(rec.output_path, "w")

        step_idx = {rec.recorder_id: 0 for rec in self.plan.recorder_plan.recorders}
        while True:
            payload = self.queue.get()
            if payload is None:
                break
                
            # Write payload to HDF5 — timed as output_writing
            t0 = _time.perf_counter_ns()
            for rec_id, data in payload.items():
                f = files[rec_id]
                ds_name = f"step_{step_idx[rec_id]}"
                f.create_dataset(ds_name, data=data)
                step_idx[rec_id] += 1
            elapsed_s = (_time.perf_counter_ns() - t0) / 1e9

            if self.profiler is not None:
                self.profiler.record("output_writing", elapsed_s)
            
        for f in files.values():
            f.close()

class Runner:
    def run(
        self,
        plan: ExecutionPlan,
        state: State,
        dt: float,
        time: float,
        chunk_size: Optional[int] = None,
        progress: bool = False,
        progress_every: int = 1,
        sync_progress: Optional[bool] = None,
        profiler=None,
    ) -> State:
        if profiler is not None:
            return self._run_profiled(plan, state, dt, time, chunk_size, progress, progress_every, profiler)
        else:
            return self._run_standard(plan, state, dt, time, chunk_size, progress, progress_every, sync_progress)

    # ==================================================================
    # Standard (production) run — unchanged from original
    # ==================================================================
    def _run_standard(
        self,
        plan: ExecutionPlan,
        state: State,
        dt: float,
        time: float,
        chunk_size: Optional[int] = None,
        progress: bool = False,
        progress_every: int = 1,
        sync_progress: Optional[bool] = None,
        ) -> State:
        total_steps = int(round(time / dt))
        if chunk_size is None:
            chunk_size = int(plan.chunk_size)
        else:
            chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        # Build step function
        # The step function is the core of the solver. It takes the current state and the current time step, and returns the functions that compute the next state. We use jax.jit to compile the step function for better performance.
        step_fn     = build_step_fn(plan, dt)
        
        # JIT compile the scan body
        @jax.jit
        def scan_chunk(s, ts):
            return jax.lax.scan(step_fn, s, ts)

        writer = AsyncWriter(plan) if plan.recorder_plan.recorders else None

        # Default: if user asked for progress, make it accurate by syncing per chunk.
        if sync_progress is None:
            sync_progress = bool(progress)

        # Robustly get base_step and base_time on host for progress and interval logic.
        def _to_int(x):
            return int(x) if isinstance(x, (int, np.integer)) else int(jax.device_get(x))

        def _to_float(x):
            return float(x) if isinstance(x, (float, int, np.floating, np.integer)) else float(jax.device_get(x))

        base_step = _to_int(state.step)
        base_time = _to_float(state.time)

        if progress:
            n_chunks = int(np.ceil(total_steps / chunk_size)) if total_steps > 0 else 0
            print(
                f"Running {total_steps} steps (dt={dt:g}, time={time:g}s) in {n_chunks} chunks (chunk_size={chunk_size})...",
                flush=True,
            )

        wall_start = _time.perf_counter()

        for chunk_idx, start in enumerate(range(0, total_steps, chunk_size)):
            n = min(chunk_size, total_steps - start)
            # Advance time by dt each step; provide the new time (t_{n+1}) to the integrator.
            ts = state.time + dt * (jnp.arange(n, dtype=jnp.float32) + 1.0)
            state, rec_buf = scan_chunk(state, ts)

            # If there are no recorders, the loop can run ahead due to JAX async dispatch.
            # Sync here only when requested (progress/accuracy).
            if sync_progress and writer is None:
                jax.block_until_ready(state.time)

            if writer is not None:
                # Async copy to host (only if we have recorders)
                host_buf = jax.device_get(rec_buf)

                # Apply recorder sampling intervals on the host.
                step_numbers = base_step + start + (np.arange(n, dtype=np.int64) + 1)
                filtered = {}
                for rec in plan.recorder_plan.recorders:
                    data = host_buf.get(rec.recorder_id)
                    if data is None:
                        continue

                    interval = int(rec.interval) if int(rec.interval) > 0 else 1
                    if interval > 1:
                        mask = (step_numbers % interval) == 0
                        data = data[mask]

                    if data.shape[0] == 0:
                        continue

                    filtered[rec.recorder_id] = data

                if filtered:
                    writer.enqueue(filtered)

            if progress:
                is_last = (start + n) >= total_steps
                if is_last or (progress_every > 0 and (chunk_idx % progress_every) == 0):
                    done_steps = start + n
                    frac = (done_steps / total_steps) if total_steps > 0 else 1.0
                    wall_now = _time.perf_counter()
                    elapsed = wall_now - wall_start
                    rate = (done_steps / elapsed) if elapsed > 0 else 0.0
                    eta = ((total_steps - done_steps) / rate) if rate > 0 else float("inf")
                    sim_t = base_time + done_steps * dt
                    print(
                        f"[{frac*100:6.2f}%] step {done_steps}/{total_steps}  t={sim_t:.6g}s  {rate:,.0f} steps/s  ETA {eta:,.1f}s",
                        flush=True,
                    )

        if writer is not None:
            writer.close()
        return state

    # ==================================================================
    # Profiled run — decomposed stepping with per-phase timing
    # ==================================================================
    def _run_profiled(
        self,
        plan: ExecutionPlan,
        state: State,
        dt: float,
        time: float,
        chunk_size: Optional[int] = None,
        progress: bool = False,
        progress_every: int = 1,
        profiler=None,
        ) -> State:
        total_steps = int(round(time / dt))
        if chunk_size is None:
            chunk_size = int(plan.chunk_size)
        else:
            chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        # Build decomposed phase functions
        phases = build_profiled_step_phases(plan, dt)
        predict_fn = phases["predict_displacement"]
        bulk_fn = phases["bulk_element_kernels"]
        interface_fn = phases["interface_coupling_kernels"]
        ext_fn = phases["compute_external_forces"]
        update_fn = phases["constraint_and_update"]
        rec_fn = phases["recorder_extraction"]

        writer = AsyncWriter(plan, profiler=profiler) if plan.recorder_plan.recorders else None

        def _to_int(x):
            return int(x) if isinstance(x, (int, np.integer)) else int(jax.device_get(x))

        def _to_float(x):
            return float(x) if isinstance(x, (float, int, np.floating, np.integer)) else float(jax.device_get(x))

        base_step = _to_int(state.step)
        base_time = _to_float(state.time)

        if progress:
            print(
                f"[PROFILED] Running {total_steps} steps (dt={dt:g}, time={time:g}s) "
                f"in decomposed mode (chunk_size={chunk_size})...",
                flush=True,
            )

        wall_start = _time.perf_counter()

        for chunk_idx, start in enumerate(range(0, total_steps, chunk_size)):
            n = min(chunk_size, total_steps - start)

            # Collect recorder data for this chunk
            chunk_rec_data = {rec.recorder_id: [] for rec in plan.recorder_plan.recorders}

            for i in range(n):
                t = _to_float(state.time) + dt
                t_jax = jnp.float32(t)

                # --- Phase: predict displacement ---
                with profiler.region("explicit_update"):
                    u_pred = predict_fn(state, t_jax)
                    # block_until_ready to ensure GPU work is done
                    jax.block_until_ready(u_pred)

                # --- Phase: bulk element kernels ---
                with profiler.region("bulk_element_kernels"):
                    f_int, new_history = bulk_fn(u_pred, state.history)
                    jax.block_until_ready(f_int)

                # --- Phase: interface/coupling kernels ---
                with profiler.region("interface_coupling_kernels"):
                    f_int = interface_fn(f_int)
                    jax.block_until_ready(f_int)

                # --- Phase: external forces + constraint + update ---
                with profiler.region("constraint_enforcement"):
                    f_ext = ext_fn(state.fields, t_jax)
                    state = update_fn(state, u_pred, f_int, f_ext, t_jax)
                    jax.block_until_ready(state.time)

                # Update history
                state = state.replace(history=new_history)

                # --- Phase: recorder extraction ---
                with profiler.region("recorder_extraction"):
                    rec_payload = rec_fn(state)
                    jax.block_until_ready(rec_payload)

                # Accumulate recorder data
                for rec in plan.recorder_plan.recorders:
                    if rec.recorder_id in rec_payload:
                        chunk_rec_data[rec.recorder_id].append(rec_payload[rec.recorder_id])

            # --- Phase: async host transfer ---
            if writer is not None:
                with profiler.region("async_host_transfer"):
                    # Stack per-step data into chunk arrays and transfer to host
                    host_buf = {}
                    for rec in plan.recorder_plan.recorders:
                        data_list = chunk_rec_data[rec.recorder_id]
                        if data_list:
                            stacked = jnp.stack(data_list, axis=0)
                            host_buf[rec.recorder_id] = jax.device_get(stacked)

                # Apply sampling intervals on host
                step_numbers = base_step + start + (np.arange(n, dtype=np.int64) + 1)
                filtered = {}
                for rec in plan.recorder_plan.recorders:
                    data = host_buf.get(rec.recorder_id)
                    if data is None:
                        continue
                    interval = int(rec.interval) if int(rec.interval) > 0 else 1
                    if interval > 1:
                        mask = (step_numbers % interval) == 0
                        data = data[mask]
                    if data.shape[0] == 0:
                        continue
                    filtered[rec.recorder_id] = data

                if filtered:
                    writer.enqueue(filtered)

            if progress:
                is_last = (start + n) >= total_steps
                if is_last or (progress_every > 0 and (chunk_idx % progress_every) == 0):
                    done_steps = start + n
                    frac = (done_steps / total_steps) if total_steps > 0 else 1.0
                    wall_now = _time.perf_counter()
                    elapsed = wall_now - wall_start
                    rate = (done_steps / elapsed) if elapsed > 0 else 0.0
                    eta = ((total_steps - done_steps) / rate) if rate > 0 else float("inf")
                    sim_t = base_time + done_steps * dt
                    print(
                        f"[PROFILED] [{frac*100:6.2f}%] step {done_steps}/{total_steps}  "
                        f"t={sim_t:.6g}s  {rate:,.0f} steps/s  ETA {eta:,.1f}s",
                        flush=True,
                    )

        if writer is not None:
            writer.close()
        return state
