"""Test that example notebooks execute successfully.

When DEBUG=True, examples use reduced parameters (coarse grids, short time ranges)
for fast test execution. This test file sets DEBUG=True for all examples and
verifies they execute without errors.

Successful execution is the only criterion—output validation is not performed.
"""

import sys
from pathlib import Path

# Use non-interactive matplotlib backend for testing
import matplotlib
import pytest

matplotlib.use("Agg")


# Find all example xx_*.py files
# Ignore those without a number prefix or that are in __pycache__ directories
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
EXAMPLE_FILES = sorted(
    [
        f
        for f in EXAMPLES_DIR.glob("*.py")
        if f.name[0:2].isdigit() and not f.name.endswith("__pycache__.py")
    ]
)


@pytest.mark.parametrize("example_file", EXAMPLE_FILES, ids=lambda f: f.name)
def test_example_execution(example_file):
    """Test that an example file executes without errors in DEBUG mode.

    Verifies successful execution of example code using reduced parameters.
    When DEBUG=True in examples, grid resolution and time ranges are coarse
    for fast iteration. See _test_config for specific parameters.

    Parameters
    ----------
    example_file : Path
        Path to example .py file
    test_context : dict
        Globals dict with DEBUG=True and test config
    """
    # Read the example file
    example_code = example_file.read_text()

    # Parse and execute with test context
    try:
        # Create execution environment with test globals
        exec_globals = {"DEBUG": True}

        # Execute the example code
        exec(compile(example_code, str(example_file), "exec"), exec_globals)

    except Exception as exc:
        pytest.fail(
            f"Example {example_file.name} failed in DEBUG mode:\n"
            f"{type(exc).__name__}: {exc}\n\n"
            f"Note: Set DEBUG=False in the example file to disable fast-mode parameters."
        )
