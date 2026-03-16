import numpy as np
import jax.numpy as jnp
from femora_solver.model.model import Model

def test_beam_r_field():
    # 1. Create a simple 2-node beam
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    model = Model("beam_test")
    model.add_nodes(coords)
    model.add_material("steel", kind="LinearElastic", E=200e9, nu=0.3, rho=7800.0)
    
    # Section: A=0.01, Iy=1e-5, Iz=1e-5, J=2e-5
    sec_params = [0.01, 1e-5, 1e-5, 2e-5]
    model.add_elements([[0, 1]], family="Beam2", material="steel", section_params=sec_params)
    
    # 2. Constraints: Fix Node 0 (U and R)
    model.add_constraint("fix_u", kind="SPC", node_indices=[0], components=[0, 1, 2], field="U")
    model.add_constraint("fix_r", kind="SPC", node_indices=[0], components=[0, 1, 2], field="R")
    
    # 3. Load: Apply torque at Node 1 (R field)
    # Applying moment around X-axis
    model.add_load("torque", kind="Nodal", field="R", node_indices=[1], force=[1000.0, 0.0, 0.0])
    
    # 4. Run
    dt = 1e-5
    time = 0.01
    print(f"Running Beam Test with R field for {time}s...")
    model.run(dt=dt, time=time)
    
    # 5. Check R field
    r_field = model._state.fields["R"].u
    print(f"Node 1 Rotation (R): {r_field[1]}")
    
    if np.abs(r_field[1, 0]) > 0:
        print("Success: Beam rotated under torque!")
    else:
        print("Failure: No rotation detected in R field.")

if __name__ == "__main__":
    test_beam_r_field()
