import pytest


def test_lazy_imports_and_caching():
    import kcmt_python  # triggers kcmt_python.__getattr__

    # First access loads and caches
    Config1 = kcmt_python.Config
    from kcmt_python.config import Config as RealConfig

    assert Config1 is RealConfig
    # Second access should use cached value
    Config2 = kcmt_python.Config
    assert Config2 is RealConfig


def test_unknown_attribute_raises():
    import kcmt_python

    with pytest.raises(AttributeError):
        getattr(kcmt_python, "TotallyUnknownSymbol")
