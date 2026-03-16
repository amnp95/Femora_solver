import h5py
import numpy as np
import pyvista as pv

def post_process():
    # 1. Read Mesh
    with h5py.File("cantilever_mesh.h5", "r") as f:
        coords = f["coords"][:]
        conn = f["conn"][:]
        
    # 2. Read Displacement steps
    with h5py.File("cantilever_full.h5", "r") as f:
        step_keys = sorted([k for k in f.keys() if k.startswith("step_")], key=lambda x: int(x.split("_")[1]))
        disp_steps = []
        for k in step_keys:
            data = f[k][:]
            disp_steps.append(data[-1])  # last in chunk

    print(f"Post-processing {len(disp_steps)} steps (total {len(coords)} nodes, {len(conn)} elements)")

    # 3. Create PyVista mesh
    n_elem = conn.shape[0]
    cells = np.hstack([np.full((n_elem, 1), 8), conn]).flatten()
    cell_types = np.full(n_elem, pv.CellType.HEXAHEDRON, dtype=np.int8)
    grid = pv.UnstructuredGrid(cells, cell_types, coords)

    # 4. Visualization setup
    warp_factor = 3e4
    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie("cantilever_movie.mp4", framerate=10)
    plotter.add_mesh(grid, style="wireframe", color="black", opacity=0.2, label="Original")
    plotter.add_scalar_bar(title="Z-Displacement")
    plotter.add_title("Femora Solver: Cantilever Beam Deflection (Hex8)")
    plotter.add_legend()

    # 5. Loop over steps and write frames
    for i, disp in enumerate(disp_steps):
        grid.point_data["Displacement"] = disp
        warped = grid.warp_by_vector("Displacement", factor=warp_factor)
        plotter.add_mesh(warped, scalars="Displacement", component=2, cmap="viridis", show_edges=True, label=f"Warped (x{warp_factor:.0e})")
        plotter.write_frame()
        plotter.clear()  # Remove warped mesh for next frame
        plotter.add_mesh(grid, style="wireframe", color="black", opacity=0.2, label="Original")
        plotter.add_scalar_bar(title="Z-Displacement")
        plotter.add_title("Femora Solver: Cantilever Beam Deflection (Hex8)")
        plotter.add_legend()

    plotter.close()
    print("Successfully saved movie to cantilever_movie.mp4")

if __name__ == "__main__":
    post_process()
