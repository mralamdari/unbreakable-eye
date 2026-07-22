"""
Shared fixtures for Unbreakable Eye tests.

All tests that import from src/ need the package to be installed
(e.g. via `pip install -e .`) or PYTHONPATH to include the repo root.
"""
import os
import sys

# Ensure the repo root is on sys.path so `from src.core.config` etc. work
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
