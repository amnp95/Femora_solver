import numpy as np
import jax.numpy as jnp
from femora_solver.model.model import Model
import h5py


def estimate_fundamental_freq_fft(signal: np.ndarray, dt: float) -> float:
    """Estimate dominant (non-DC) frequency using an FFT of a windowed, de-meaned signal."""
    y = np.asarray(signal, dtype=float)
    if y.size < 8 or dt <= 0:
        return 0.0
    y = y - np.mean(y)
    window = np.hanning(y.size)
    Y = np.fft.rfft(y * window)
    freqs = np.fft.rfftfreq(y.size, d=dt)

    mag = np.abs(Y)
    if mag.size <= 1:
        return 0.0
    idx = int(np.argmax(mag[1:]) + 1)  # skip DC bin
    return float(freqs[idx])

def generate_cantilever_mesh(nx, ny, nz, lx, ly, lz):
    x = np.linspace(0, lx, nx + 1)
    y = np.linspace(0, ly, ny + 1)
    z = np.linspace(0, lz, nz + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
    def get_node_idx(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
    conn = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n1, n2, n3, n4 = get_node_idx(i,j,k), get_node_idx(i+1,j,k), get_node_idx(i+1,j+1,k), get_node_idx(i,j+1,k)
                n5, n6, n7, n8 = get_node_idx(i,j,k+1), get_node_idx(i+1,j,k+1), get_node_idx(i+1,j+1,k+1), get_node_idx(i,j+1,k+1)
                conn.append([n1, n2, n3, n4, n5, n6, n7, n8])
    return coords, conn

def run_verification():
    # Slender beam for better analytical match (L/h = 10)
    nx, ny, nz = 20, 2, 2
    lx, ly, lz = 10.0, 1.0, 1.0
    E, nu, rho = 200e9, 0.3, 7800.0
    
    coords, conn = generate_cantilever_mesh(nx, ny, nz, lx, ly, lz)
    model = Model("cantilever_slender")
    model.add_nodes(coords)
    model.add_material("steel", kind="LinearElastic", E=E, nu=nu, rho=rho)
    model.add_elements(conn, family="Hex8", material="steel")
    
    fixed_nodes = np.where(coords[:, 0] == 0)[0]
    model.add_constraint("fixed", kind="SPC", node_indices=fixed_nodes.tolist())
    
    tip_nodes = np.where(coords[:, 0] == lx)[0]
    P_total = -1.0e4 # 10kN
    P_per_node = P_total / len(tip_nodes)
    force = np.zeros((len(tip_nodes), 3))
    force[:, 2] = P_per_node
    model.add_load("tip_load", kind="Nodal", field="U", node_indices=tip_nodes.tolist(), force=force.tolist())

    # Recorder: capture tip displacement history so we can measure true peaks over time
    h5_file = "cantilever_history.h5"
    rec_interval = 50
    model.add_recorder(
        "tip",
        kind="NodeSet",
        node_indices=tip_nodes.tolist(),
        field="U",
        interval=rec_interval,
        file=h5_file,
    )
    
    # Run long enough to capture several cycles (fundamental period is ~0.1s for this geometry)
    dt = 5e-6
    time = 0.5
    print(f"Running cantilever simulation for {time}s (dt={dt}, rec_interval={rec_interval})...")
    model.run(dt=dt, time=time)

    # Final snapshot (phase-dependent in undamped dynamics)
    u_tip_final = np.asarray(model._state.fields["U"].u[tip_nodes])
    final_min_z = float(np.min(u_tip_final[:, 2]))
    final_avg_z = float(np.mean(u_tip_final[:, 2]))

    # Time history (true peak over time)
    with h5py.File(h5_file, "r") as f:
        keys = sorted(f.keys(), key=lambda x: int(x.split('_')[1]))
        if not keys:
            raise RuntimeError(
                "Recorder produced no data. Reduce rec_interval or increase simulation time."
            )
        chunks = [f[k][:] for k in keys]  # (n_rec_in_chunk, n_tip, 3)
        tip_u_hist = np.concatenate(chunks, axis=0)

    dt_rec = dt * rec_interval
    tip_z_min_hist = tip_u_hist[:, :, 2].min(axis=1)
    tip_z_avg_hist = tip_u_hist[:, :, 2].mean(axis=1)
    peak_min_z = float(np.min(tip_z_min_hist))
    peak_avg_z = float(np.min(tip_z_avg_hist))
    tail_start = int(0.8 * len(tip_z_avg_hist))
    tail_mean_avg_z = float(np.mean(tip_z_avg_hist[tail_start:]))
    
    # Analytical Static
    I = (ly * lz**3) / 12.0
    delta_static = (P_total * lx**3) / (3 * E * I)
    # Transient peak should be approx 2x static
    expected_peak = 2.0 * delta_static

    # Theoretical fundamental frequency (Euler-Bernoulli cantilever, mode 1)
    A = ly * lz
    freq_theory = (3.516 / (2 * np.pi * lx**2)) * np.sqrt((E * I) / (rho * A))

    freq_num = estimate_fundamental_freq_fft(tip_z_avg_hist, dt_rec)
    
    print("\n--- CANTILEVER METRICS (Z tip displacement) ---")
    print(f"Analytical Static Displacement:        {delta_static:.6e} m")
    print(f"Expected Dynamic Peak (~2x static):    {expected_peak:.6e} m")
    print("\nTime-history metrics (from recorder):")
    print(f"Peak over time (min of tip avg):       {peak_avg_z:.6e} m")
    print(f"Peak over time (min of tip nodes):     {peak_min_z:.6e} m")
    print(f"Tail-mean (last 20%, tip avg):         {tail_mean_avg_z:.6e} m")
    print(f"Ratio (Peak avg / Static):             {peak_avg_z / delta_static:.4f}")
    print(f"Ratio (Peak avg / Expected):           {peak_avg_z / expected_peak:.4f}")
    print(f"Ratio (Tail-mean / Static):            {tail_mean_avg_z / delta_static:.4f}")
    print("\nFinal snapshot metrics (phase-dependent):")
    print(f"Final (min of tip nodes):              {final_min_z:.6e} m")
    print(f"Final (avg of tip nodes):              {final_avg_z:.6e} m")
    print("\nFrequency:")
    print(f"Theoretical fundamental freq:          {freq_theory:.2f} Hz")
    print(f"Numerical freq (FFT):                  {freq_num:.2f} Hz")

if __name__ == "__main__":
    run_verification()
