"""Verify the development environment is correctly configured."""

import sys


def test_python_version():
    """Ensure Python 3.10+ is available."""
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version}"


def test_planned_directories_exist():
    """Verify the planned project directory structure is in place."""
    from pathlib import Path

    root = Path(__file__).parent.parent
    expected_dirs = [
        "infra/ansible",
        "infra/slurm",
        "kernels",
        "training/src",
        "training/recipes",
        "training/slurm",
        "training/evaluation",
        "serving/k8s",
        "serving/vllm_config",
        "serving/monitoring",
        "tests",
        "memory-bank",
    ]
    for d in expected_dirs:
        assert (root / d).is_dir(), f"Missing directory: {d}"


def test_pyproject_toml_exists():
    """Ensure pyproject.toml is present and parseable."""
    from pathlib import Path

    root = Path(__file__).parent.parent
    pyproject = root / "pyproject.toml"
    assert pyproject.is_file(), "pyproject.toml not found"
    content = pyproject.read_text()
    assert "infinitecontext-1b" in content
