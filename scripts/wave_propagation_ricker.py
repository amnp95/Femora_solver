import numpy as np
from femora_solver.model.model import Model
import h5py


def generate_hex_mesh(nx: int, ny: int, nz: int, lx: float, ly: float, lz: float, dtype=np.float32):
    """Vectorized regular Hex8 mesh.

    Returns:
      coords: (n_nodes, 3) float32
      conn:   (n_elem, 8) int64
    """
    x = np.linspace(0.0, lx, nx + 1, dtype=dtype)
    y = np.linspace(0.0, ly, ny + 1, dtype=dtype)
    z = np.linspace(0.0, lz, nz + 1, dtype=dtype)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    coords = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=-1)

    ny1 = ny + 1
    nz1 = nz + 1

    i = np.arange(nx, dtype=np.int64)[:, None, None]
    j = np.arange(ny, dtype=np.int64)[None, :, None]
    k = np.arange(nz, dtype=np.int64)[None, None, :]

    base = i * (ny1 * nz1) + j * nz1 + k

    n0 = base
    n1 = base + (ny1 * nz1)
    n3 = base + nz1
    n2 = n1 + nz1

    n4 = base + 1
    n5 = n1 + 1
    n7 = n3 + 1
    n6 = n2 + 1

    conn = np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=-1).reshape(-1, 8)
    return coords, conn


def run_wave_ricker_box():
    # ---- Mesh / material ----
    # Scale up to 100x100x100 for 1M Hex8 elements (101^3 ~ 1.03M nodes)
    nx, ny, nz = 30, 30, 30
    lx, ly, lz = 10.0, 10.0, 10.0

    E, nu, rho = 200e9, 0.3, 7800.0

    coords, conn = generate_hex_mesh(nx, ny, nz, lx, ly, lz)

    mesh_file = "wave_ricker_mesh.h5"
    with h5py.File(mesh_file, "w") as f:
        f.create_dataset("coords", data=coords)
        f.create_dataset("conn", data=conn)

    model = Model("wave_ricker_box")
    model.add_nodes(coords)
    model.add_material("steel", kind="LinearElastic", E=E, nu=nu, rho=rho)
    model.add_elements(conn, family="Hex8", material="steel")

    # ---- Boundary conditions: fix 5 faces (left/right/front/back/bottom) ----
    tol = 0.0  # regular grid -> exact comparisons are ok
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    left = np.where(x == 0.0 + tol)[0]
    right = np.where(x == lx - tol)[0]
    front = np.where(y == 0.0 + tol)[0]
    back = np.where(y == ly - tol)[0]
    bottom = np.where(z == 0.0 + tol)[0]

    fixed_nodes = np.unique(np.concatenate([left, right, front, back, bottom]))
    model.add_constraint(
        "fixed_box",
        kind="SPC",
        node_indices=fixed_nodes.tolist(),
        components=[0, 1, 2],
        field="U",
    )

    # ---- Load: Ricker wavelet at the center node ----
    ic, jc, kc = nx // 2, ny // 2, nz // 2
    center_node = ic * (ny + 1) * (nz + 1) + jc * (nz + 1) + kc

    # Peak force magnitude (N) applied in Z
    F0 = -1.0e4

    # Ricker parameters
    f0 = 50.0   # Hz
    t0 = 1.0 / f0  # peak time (s)

    model.add_load(
        "ricker_center",
        kind="Nodal",
        field="U",
        node_indices=[int(center_node)],
        force=[0.0, 0.0, F0],
        time_fn={"kind": "Ricker", "f0": f0, "t0": t0, "amp": 1.0},
    )

    # ---- Recorders: probe line along +Z at (x=mid,y=mid) ----
    probe_nodes = [
        ic * (ny + 1) * (nz + 1) + jc * (nz + 1) + kk
        for kk in range(nz + 1)
    ]

    rec_file = "wave_ricker_probes.h5"
    rec_interval = 10
    model.add_recorder(
        "probes",
        kind="NodeSet",
        node_indices=[int(n) for n in probe_nodes],
        field="U",
        interval=rec_interval,
        file=rec_file,
    )

    field_file = "wave_ricker_field.h5"
    model.add_recorder(
        "field",
        kind="NodeSet",
        node_indices=np.arange(coords.shape[0], dtype=np.int64),
        field="U",
        interval=rec_interval,
        file=field_file,
    )

    # ---- Run ----
    dt = 1e-5
    time = 0.02
    print(f"Running wave propagation: time={time}s dt={dt} (rec_interval={rec_interval})")
    model.run(dt=dt, time=time, progress=True)

    # ---- Quick post metrics: arrival at bottom probe ----
    with h5py.File(rec_file, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        buf = np.concatenate([f[k][:] for k in keys], axis=0)  # (n_samples, n_probe, 3)

    dt_rec = dt * rec_interval
    t = np.arange(buf.shape[0]) * dt_rec

    z_center = buf[:, kc, 2]  # displacement at center node (within probe line)
    z_bottom = buf[:, 0, 2]

    # crude arrival time: first time bottom exceeds 1% of its peak magnitude
    peak = np.max(np.abs(z_bottom))
    thr = 0.01 * peak if peak > 0 else 0.0
    idx = np.argmax(np.abs(z_bottom) >= thr) if thr > 0 else 0

    print("\n--- QUICK METRICS ---")
    print(f"Center node peak |Uz|:  {np.max(np.abs(z_center)):.6e} m")
    print(f"Bottom node peak |Uz|:  {np.max(np.abs(z_bottom)):.6e} m")
    if thr > 0:
        print(f"Bottom arrival (>=1% peak): t ~ {t[idx]:.6e} s")
    else:
        print("Bottom arrival: (no signal detected)")


if __name__ == "__main__":
    run_wave_ricker_box()
