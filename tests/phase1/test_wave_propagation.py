import numpy as np
import pytest
from femora_solver.model.model import Model

def test_wave_propagation():
    model = Model("wave")
    
    # 1 element cube
    coords = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ], dtype=np.float32)
    
    model.add_nodes(coords)
    model.add_material("steel", kind="LinearElastic", E=200e9, nu=0.3, rho=7800.0)
    model.add_elements([[0,1,2,3,4,5,6,7]], family="Hex8", material="steel")
    
    model.add_constraint("fixed_base", kind="SPC", node_indices=[0,1,2,3], components=[0,1,2])
    model.add_load("impulse", kind="Nodal", field="U", node_indices=[4,5,6,7], force=[0.0, 0.0, -1000.0])
    
    model.add_recorder("disp", kind="NodeSet", node_indices=[4,5,6,7], field="U", interval=10, file="test_out.h5")
    
    model.run(dt=1e-5, time=1e-4) # 10 steps
    
    # Check that state progressed
    assert model._state.step == 10
    assert np.isclose(model._state.time, 1e-4)
