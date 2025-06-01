"""Basic tests to ensure package is working."""

import pytest
from mostlyaiprize import __version__


def test_version():
    """Test that version is accessible."""
    assert __version__ is not None
    assert isinstance(__version__, str)


def test_import():
    """Test that package can be imported."""
    import mostlyaiprize  # noqa: F401
