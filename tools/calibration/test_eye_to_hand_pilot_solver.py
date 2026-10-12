import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("eye_to_hand_pilot_solver.py")
SPEC = importlib.util.spec_from_file_location("eye_to_hand_pilot_solver", MODULE_PATH)
SOLVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SOLVER)


def test_synthetic_data_recovers_frozen_transform_directions():
    result = SOLVER.run_synthetic_direction_test()
    assert result["passed"] is True
    assert result["equation"] == "C_T_Tag_i = C_T_B * B_T_H_i * H_T_Tag"
    assert result["quaternion_order"] == "XYZW"
    assert set(result["methods"]) == set(SOLVER.METHODS)
    assert all(method["passed"] for method in result["methods"].values())
