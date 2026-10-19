"""Keep local validator installs aligned with the image's dependency source."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]


def test_root_requirements_do_not_pin_sdk():
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            assert canonicalize_name(Requirement(line).name) != "oro-sdk", (
                "Declare oro-sdk only in docker/validator/pyproject.toml"
            )


def test_validator_sdk_pin_matches_lockfile():
    validator = ROOT / "docker" / "validator"
    project = tomllib.loads((validator / "pyproject.toml").read_text())
    lock = tomllib.loads((validator / "uv.lock").read_text())
    sdk_requirements = [
        Requirement(dep)
        for dep in project["project"]["dependencies"]
        if canonicalize_name(Requirement(dep).name) == "oro-sdk"
    ]
    sdk_versions = [
        pkg["version"] for pkg in lock["package"] if pkg["name"] == "oro-sdk"
    ]
    assert len(sdk_requirements) == len(sdk_versions) == 1
    assert str(sdk_requirements[0].specifier) == f"=={sdk_versions[0]}"
