def test_config_importable():
    import config  # noqa: F401


def test_palette_importable():
    import palette  # noqa: F401


def test_renderer_importable():
    import renderer  # noqa: F401


def test_no_circular_import():
    """Import order matches DAG: config has no deps; palette and renderer depend on config only."""
    import config
    import palette
    import renderer
    # If this doesn't raise, no circular import occurred
