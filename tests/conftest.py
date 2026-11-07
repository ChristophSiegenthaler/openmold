import shutil
import pytest


def pytest_collection_modifyitems(config, items):
    if shutil.which("ccx"):
        return
    skip = pytest.mark.skip(reason="ccx not installed")
    for item in items:
        if "solver" in item.keywords:
            item.add_marker(skip)
