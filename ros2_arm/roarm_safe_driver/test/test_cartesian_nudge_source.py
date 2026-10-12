import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
NODE = PACKAGE / "roarm_safe_driver" / "node.py"


def test_legacy_cartesian_nudge_surface_is_not_exposed():
    """The maintained paper path has one joint-target command surface only."""

    source = NODE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "_cartesian_nudge_callback" not in methods
    assert "/roarm_safe/cartesian_nudge" not in source
    assert "command_cartesian_nudge" not in source
    assert '"/roarm_safe/joint_target"' in source
