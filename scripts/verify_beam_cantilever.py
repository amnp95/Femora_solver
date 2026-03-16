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

def run_beam_verification():
    # Slender beam (L/h = 10)
    nx = 20
    lx, ly, lz = 10.0, 1.0, 1.0
    E, nu, rho = 200e9, 0.3, 7800.0
    
    coords = np.zeros((nx + 1, 3))
    coords[:, 0] = np.linspace(0, lx, nx + 1)
    conn = [[i, i + 1] for i in range(nx)]
    
    model = Model("beam_cantilever")
    model.add_nodes(coords)
    model.add_material("steel", kind="LinearElastic", E=E, nu=nu, rho=rho)
    
    A = ly * lz
    Iz = (lz * ly**3) / 12.0
    Iy = (ly * lz**3) / 12.0
    J = Iy + Iz
    model.add_elements(conn, family="Beam2", material="steel", section_params=[A, Iy, Iz, J])
    
    model.add_constraint("fixed_u", kind="SPC", node_indices=[0], field="U")
    model.add_constraint("fixed_r", kind="SPC", node_indices=[0], field="R")
    
    P_total = -1.0e4
    model.add_load("tip_load", kind="Nodal", field="U", node_indices=[nx], force=[0.0, 0.0, P_total])
    
    # Use a recorder to capture history
    h5_file = "beam_history.h5"
    rec_interval = 10
    model.add_recorder("tip", kind="NodeSet", node_indices=[nx], field="U", interval=rec_interval, file=h5_file)
    
    dt = 1e-6
    time = 1.5 # Enough for ~5 cycles
    print(f"Running Beam simulation for {time}s (dt={dt}, rec_interval={rec_interval})...")
    model.run(dt=dt, time=time)
    
    # Post-process history
    with h5py.File(h5_file, "r") as f:
        # Collect all steps
        disps = []
        for key in sorted(f.keys(), key=lambda x: int(x.split('_')[1])):
            disps.append(f[key][:]) # shape (chunk, 1, 3)
        history = np.concatenate(disps, axis=0)[:, 0, 2] # Z-disp history

    dt_rec = dt * rec_interval
        
    peak_dyn = np.min(history)
    avg_dyn = np.mean(history)
    final_dyn = history[-1]
    
    # Analytical Calculations
    delta_static = (P_total * lx**3) / (3 * E * Iz)
    
    # Theoretical frequency: f = (3.516 / 2pi*L^2) * sqrt(EI/rhoA)
    freq_theory = (3.516 / (2 * np.pi * lx**2)) * np.sqrt((E * Iz) / (rho * A))
    
    # Numerical frequency: count zero crossings or peaks
    freq_num = estimate_fundamental_freq_fft(history, dt_rec)
    
    print("\n--- BEAM CANTILEVER METRICS ---")
    print(f"Analytical Static Displacement:   {delta_static:.6e} m")
    print(f"Solver Average Displacement:      {avg_dyn:.6e} m")
    print(f"Solver Peak Displacement:         {peak_dyn:.6e} m")
    print(f"Solver Final Displacement:        {final_dyn:.6e} m")
    print(f"Ratio (Average / Static):         {avg_dyn / delta_static:.4f}")
    print(f"Ratio (Peak / Static):            {peak_dyn / delta_static:.4f} (Theoretical Target: 2.0)")
    print("-" * 31)
    print(f"Theoretical Fundamental Freq:     {freq_theory:.2f} Hz")
    print(f"Solver Numerical Frequency:       {freq_num:.2f} Hz")
    print(f"Frequency Accuracy:               {freq_num / freq_theory:.4f}")

if __name__ == "__main__":
    run_beam_verification()
