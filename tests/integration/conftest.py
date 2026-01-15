"""Integration tests configuration - these tests require database and external services."""

import pytest

# Mark all tests in this folder as integration tests
def pytest_collection_modifyitems(items):
    """Mark all tests in this folder with 'integration' marker."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
