"""Lazy public API, optional data extra and version consistency."""
from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import ramen_foundry

ROOT = Path(__file__).resolve().parents[1]


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter so import side effects are observable."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def test_importing_the_package_does_not_import_langgraph() -> None:
    result = run_python(
        """
        import sys
        import ramen_foundry
        loaded = [m for m in ("langgraph", "langchain_core", "ramen_foundry.core", "ramen_foundry.templates") if m in sys.modules]
        assert not loaded, loaded
        """
    )
    assert result.returncode == 0, result.stderr


def test_the_data_module_does_not_import_langgraph() -> None:
    pytest.importorskip("pandas")
    result = run_python(
        """
        import sys
        import ramen_foundry.data
        loaded = [m for m in ("langgraph", "langchain_core", "ramen_foundry.core") if m in sys.modules]
        assert not loaded, loaded
        """
    )
    assert result.returncode == 0, result.stderr


def test_the_data_module_explains_how_to_install_pandas() -> None:
    result = run_python(
        """
        import sys
        sys.modules["pandas"] = None  # makes `import pandas` raise ImportError
        try:
            import ramen_foundry.data
        except ImportError as error:
            print(error)
        else:
            raise SystemExit("ImportError was not raised")
        """
    )
    assert result.returncode == 0, result.stderr
    assert 'pip install "ramen-foundry[data]"' in result.stdout


def test_existing_quickstart_imports_still_work() -> None:
    from ramen_foundry import DevboxShieldAgent, RamenToolNode, RemoteForgeMemoryStore

    assert DevboxShieldAgent.__module__.startswith("ramen_foundry.templates")
    assert RamenToolNode.__module__.startswith("ramen_foundry.core")
    assert RemoteForgeMemoryStore.__name__ == "RemoteForgeMemoryStore"


def test_every_name_in_all_resolves_and_is_listed_by_dir() -> None:
    listed = dir(ramen_foundry)
    for name in ramen_foundry.__all__:
        assert getattr(ramen_foundry, name) is not None, name
        assert name in listed, name
    assert len(ramen_foundry.__all__) == len(set(ramen_foundry.__all__))


def test_a_lazy_name_is_the_same_object_as_its_defining_module_attribute() -> None:
    from ramen_foundry import core, templates

    assert ramen_foundry.RamenToolNode is core.RamenToolNode
    assert ramen_foundry.ReceiptEvidenceVerifier is templates.ReceiptEvidenceVerifier


def test_an_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'DoesNotExist'"):
        ramen_foundry.DoesNotExist  # noqa: B018


def test_the_package_version_matches_pyproject() -> None:
    declared = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)
    assert declared is not None
    assert ramen_foundry.__version__ == declared.group(1)


def test_the_pyproject_pins_the_expected_extras() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^data = \[\s*"pandas>=2\.2\.3,<3\.0\.0",?\s*\]', text, re.M)
    assert '"ramen-ai-core>=0.3.3,<0.5.0"' in text
    assert "langgraph" in text  # still a default dependency
