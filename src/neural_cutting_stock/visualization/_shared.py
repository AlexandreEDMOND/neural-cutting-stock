"""Internal formatting helpers shared by the phase publication modules."""


def seconds(value: float | None) -> str:
    """Render a measured duration for publication tables."""

    return "n/a" if value is None else f"{value:.6f}"


def number(value: float | None) -> str:
    """Render a measured quantity for publication tables."""

    return "n/a" if value is None else f"{value:g}"
