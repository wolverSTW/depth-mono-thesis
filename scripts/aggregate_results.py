"""Convenience entry point for the shared real-evidence aggregator."""
import runpy
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.argv.insert(1, "aggregate")
runpy.run_path(str(Path(__file__).with_name("research.py")), run_name="__main__")
