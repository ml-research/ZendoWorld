"""Locations of bundled resources and user-owned project files."""
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
_SOURCE_ROOT = PACKAGE_ROOT.parents[1]
PROJECT_ROOT = Path(os.environ.get("ZENDOWORLD_HOME", str(_SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").exists() else Path.cwd()))).expanduser().resolve()
CONFIG_DIR = PROJECT_ROOT / "configs"
CREDENTIALS_DIR = CONFIG_DIR / "credentials"
VISION_WEIGHTS = PROJECT_ROOT / "weights" / "vision" / "zendo_model.pt"
RULES_FILE = PACKAGE_ROOT / "rules" / "rules.pl"
RENDER_CONFIG = PACKAGE_ROOT / "rendering" / "configs" / "simple_config.yml"
RENDER_ASSETS = PACKAGE_ROOT / "rendering" / "assets"


def output_path(*parts):
    return PROJECT_ROOT.joinpath("outputs", *parts)
