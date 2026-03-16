import os

# Windows + HDF5 can hit file-locking errors (Win32 GetLastError=33) even for
# read-only access. Disabling HDF5 file locking is safe here because we only
# read finalized recorder outputs.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pyvista as pv


def _sorted_step_keys(h5: h5py.File):
    return sorted([k for k in h5.keys() if k.startswith("step_")], key=lambda x: int(x.split("_")[1]))


def _h5_open(path: str, mode: str):
    try:
        return h5py.File(path, mode, locking=False)
    except TypeError:
        return h5py.File(path, mode)


def _compute_global_max_u_mag(field_file: str) -> float:
    max_mag = 0.0
    with _h5_open(field_file, "r") as f:
        for k in _sorted_step_keys(f):
            data = f[k][:]  # (n_samples, n_nodes, 3)
            if data.size == 0:
                continue
            mag = np.linalg.norm(data, axis=2)
            max_mag = max(max_mag, float(np.max(mag)))
    return max_mag


def make_wave_movie(
    mesh_file: str = "wave_ricker_mesh.h5",
    field_file: str = "wave_ricker_field.h5",
    out_movie: str = "wave_ricker_movie.mp4",
    fps: int = 24,
    warp_ratio: float = 1.0,
    off_screen: bool = False,
):
    with _h5_open(mesh_file, "r") as f:
        coords = f["coords"][:]
        conn = f["conn"][:]

    n_elem = conn.shape[0]
    cells = np.hstack([np.full((n_elem, 1), 8, dtype=np.int64), conn.astype(np.int64)]).ravel()
    cell_types = np.full(n_elem, pv.CellType.HEXAHEDRON, dtype=np.int8)
    grid = pv.UnstructuredGrid(cells, cell_types, coords)
    plotter = pv.Plotter()
    plotter.open_movie(out_movie, framerate=fps)
    plotter.add_mesh(grid, show_edges=True, opacity=1)
    coords = grid.points.copy()

    with _h5_open(field_file, "r") as f:
        keys = _sorted_step_keys(f)
        if not keys:
            raise RuntimeError(f"No step datasets found in {field_file!r}.")

        for k in keys:
            buf = f[k][:]  # (n_samples, n_nodes, 3)
            if buf.size == 0:
                continue

            for i in range(buf.shape[0]):
                u = buf[i]
                grid.points = coords + u * 1e9
                plotter.render()
                plotter.write_frame()

    plotter.close()
    print(f"Saved movie: {out_movie}")


if __name__ == "__main__":
    make_wave_movie()
