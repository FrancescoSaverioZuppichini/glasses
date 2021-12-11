import pytest
from pathlib import Path


@pytest.fixture
def glasses_path(tmp_path: Path) -> Path:
    return tmp_path / "glasses"
