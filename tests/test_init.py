import pytest


def test_lazy_imports_and_caching():
    import kcmt  # triggers kcmt.__getattr__

    # First access loads and caches
    Config1 = kcmt.Config
    from kcmt.config import Config as RealConfig

    assert Config1 is RealConfig
    # Second access should use cached value
    Config2 = kcmt.Config
    assert Config2 is RealConfig


def test_unknown_attribute_raises():
    import kcmt

    with pytest.raises(AttributeError):
        getattr(kcmt, "TotallyUnknownSymbol")
