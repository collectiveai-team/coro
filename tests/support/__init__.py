"""Importable test helpers shared across the mirrored test tree.

`conftest.py` holds fixtures — anything with per-test setup or teardown. This
package holds the pure factories, constants and stand-ins that test modules
need at import time, so they are reached by a normal import from any depth
rather than by importing `conftest` as a module.

Resolved via `pythonpath = ["tests"]` in `[tool.pytest.ini_options]`.
"""
