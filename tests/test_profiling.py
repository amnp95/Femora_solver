"""Tests for the SolverProfiler and profiled execution mode."""

import json
import os
import tempfile

import numpy as np
import pytest

from femora_solver.model.model import Model
from femora_solver.profiling.timer import SolverProfiler, PHASE_NAMES


# ---- Unit tests for SolverProfiler ----

class TestSolverProfiler:
    def test_region_timing(self):
        p = SolverProfiler()
        with p.region("test_phase"):
            pass  # near-zero duration
        assert "test_phase" in p._phases
        assert p._phases["test_phase"].count == 1
        assert p._phases["test_phase"].total_s >= 0.0

    def test_record_manual(self):
        p = SolverProfiler()
        p.record("my_phase", 0.123)
        p.record("my_phase", 0.456)
        assert p._phases["my_phase"].count == 2
        assert abs(p._phases["my_phase"].total_s - 0.579) < 1e-9

    def test_summary_output(self):
        p = SolverProfiler()
        p.record("bulk_element_kernels", 1.5)
        p.record("constraint_enforcement", 0.3)
        summary = p.summary()
        assert "bulk_element_kernels" in summary
        assert "constraint_enforcement" in summary
        assert "Total accounted" in summary

    def test_to_dict(self):
        p = SolverProfiler()
        p.record("bulk_element_kernels", 1.0)
        d = p.to_dict()
        assert "meta" in d
        assert "phases" in d
        assert "bulk_element_kernels" in d["phases"]
        assert d["phases"]["bulk_element_kernels"]["count"] == 1

    def test_to_json(self):
        p = SolverProfiler()
        p.record("explicit_update", 0.5)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            p.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert "phases" in data
            assert "explicit_update" in data["phases"]
        finally:
            os.unlink(path)

    def test_reset(self):
        p = SolverProfiler()
        p.record("test", 1.0)
        p.mark_start()
        p.mark_end()
        p.reset()
        assert len(p._phases) == 0
        assert p.total_wall_s == 0.0

    def test_wall_clock(self):
        p = SolverProfiler()
        p.mark_start()
        import time; time.sleep(0.01)
        p.mark_end()
        assert p.total_wall_s > 0.0


# ---- Integration test: profiled run ----

class TestProfiledRun:
    def _build_simple_model(self):
        """Build the same 1-element model from test_wave_propagation."""
        model = Model("profile_test")
        coords = np.array([
            [0,0,0], [1,0,0], [1,1,0], [0,1,0],
            [0,0,1], [1,0,1], [1,1,1], [0,1,1]
        ], dtype=np.float32)
        model.add_nodes(coords)
        model.add_material("steel", kind="LinearElastic", E=200e9, nu=0.3, rho=7800.0)
        model.add_elements([[0,1,2,3,4,5,6,7]], family="Hex8", material="steel")
        model.add_constraint("fix", kind="SPC", node_indices=[0,1,2,3], components=[0,1,2])
        model.add_load("imp", kind="Nodal", field="U", node_indices=[4,5,6,7],
                        force=[0.0, 0.0, -1000.0])
        model.add_recorder("disp", kind="NodeSet", node_indices=[4,5,6,7],
                           field="U", interval=1, file="profile_test_out.h5")
        return model

    def test_profiled_mode_captures_phases(self):
        """Verify that profiled mode captures timing for all expected phases."""
        model = self._build_simple_model()
        model.run(dt=1e-5, time=5e-5, profile=True)  # 5 steps

        assert model._profiler is not None
        profiler = model._profiler

        # Check that the key solver phases were recorded
        expected_phases = {
            "model_compilation",
            "field_layout_construction",
            "partitioning",
            "bulk_element_kernels",
            "constraint_enforcement",
            "explicit_update",
            "recorder_extraction",
            "async_host_transfer",
        }
        recorded = set(profiler._phases.keys())
        missing = expected_phases - recorded
        assert not missing, f"Missing profiled phases: {missing}"

        # All timings should be non-negative
        for name, phase in profiler._phases.items():
            for dur in phase.durations_s:
                assert dur >= 0.0, f"Negative duration in {name}: {dur}"

    def test_profiled_mode_state_progression(self):
        """Verify that profiled mode produces correct simulation state."""
        model = self._build_simple_model()
        model.run(dt=1e-5, time=1e-4, profile=True)  # 10 steps

        assert model._state.step == 10
        assert np.isclose(float(model._state.time), 1e-4, rtol=1e-3)

    def test_profiled_json_export(self):
        """Verify JSON export works from a profiled run."""
        model = self._build_simple_model()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
        try:
            model.run(dt=1e-5, time=5e-5, profile=True, profile_export=json_path)
            with open(json_path) as f:
                data = json.load(f)
            assert "meta" in data
            assert "phases" in data
            assert len(data["phases"]) > 0
        finally:
            os.unlink(json_path)

    def test_profiled_vs_standard_consistency(self):
        """Verify profiled mode gives same results as standard mode."""
        # Standard run
        model_std = self._build_simple_model()
        model_std._recorders = []  # Remove recorder for clean comparison
        model_std._dirty = 4
        model_std.run(dt=1e-5, time=5e-5)
        u_std = np.asarray(model_std._state.fields["U"].u)

        # Profiled run
        model_prof = self._build_simple_model()
        model_prof._recorders = []
        model_prof._dirty = 4
        model_prof.run(dt=1e-5, time=5e-5, profile=True)
        u_prof = np.asarray(model_prof._state.fields["U"].u)

        np.testing.assert_allclose(u_prof, u_std, rtol=1e-5, atol=1e-12,
                                   err_msg="Profiled and standard modes diverged")

    def test_summary_format(self):
        """Verify the summary string is well-formed."""
        model = self._build_simple_model()
        model.run(dt=1e-5, time=5e-5, profile=True)
        summary = model._profiler.summary()
        assert "Femora Solver Profile" in summary
        assert "bulk_element_kernels" in summary
        assert "%Wall" in summary


# Cleanup generated files after tests
@pytest.fixture(autouse=True)
def cleanup_files():
    yield
    for f in ["profile_test_out.h5", "test_out.h5"]:
        if os.path.exists(f):
            os.unlink(f)
