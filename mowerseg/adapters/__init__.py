"""Adapter entrypoints (CLI / FastAPI). Core does not depend on these."""

from mowerseg.cli import main as cli_main
from mowerseg.server import app

__all__ = ["app", "cli_main"]
