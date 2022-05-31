"""
Unit and regression test for the clustering package.
"""

# Import package, test suite, and other packages as needed
import clustering
import pytest
import sys

def test_templatecorr_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "clustering" in sys.modules
