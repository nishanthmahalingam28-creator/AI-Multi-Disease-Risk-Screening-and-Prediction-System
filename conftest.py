"""Pytest configuration for test suite isolation.

Prevents module namespace collisions between sibling disease models that share
bare import names (e.g. `predict` and `preprocessing`).
"""

import sys
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    """Purge cached top-level sibling model modules before module collection."""
    for mod_name in ["predict", "preprocessing"]:
        sys.modules.pop(mod_name, None)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Purge cached top-level sibling model modules before test execution."""
    for mod_name in ["predict", "preprocessing"]:
        sys.modules.pop(mod_name, None)
