import queue
import threading
import jax
import h5py
import numpy as np
import time as _time
from typing import Optional
from femora_solver.state.state import State
from femora_solver.compile.execution_plan import ExecutionPlan
from femora_solver.analysis.transient_explicit import build_step_fn
import jax.numpy as jnp

class AsyncWriter:
    def __init__(self, plan: ExecutionPlan):
        self.plan = plan
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
                
            # Write payload to HDF5
            for rec_id, data in payload.items():
                f = files[rec_id]
                ds_name = f"step_{step_idx[rec_id]}"
                f.create_dataset(ds_name, data=data)
                step_idx[rec_id] += 1
            
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
    ) -> State:
        total_steps = int(round(time / dt))
        if chunk_size is None:
            chunk_size = int(plan.chunk_size)
        else:
            chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        step_fn     = build_step_fn(plan, dt)
        
        # JIT compile the scan body
        # Actually, lax.scan itself handles this well, but we can compile the scan loop
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
                # Each scan output row corresponds to global step: base_step + start + (i + 1)
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
