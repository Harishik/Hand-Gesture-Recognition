"""The browser app's JavaScript features and MLP must match Python (fixture written by scripts/export_web.py)."""

import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
@pytest.mark.skipif(not (WEB / "models" / "parity.json").exists(), reason="run scripts/export_web.py first")
def test_javascript_matches_python():
    result = subprocess.run(["node", str(WEB / "tests" / "parity.mjs")], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
