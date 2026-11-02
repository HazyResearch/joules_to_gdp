"""GDPVal dataset provider for Joules to GDP runs."""

from .base import DatasetProvider


def ensure_registered() -> None:
    """Register the GDPVal dataset and its evaluation handlers."""
    import ipw.evaluation  # noqa: F401

    from . import gdpval  # noqa: F401


__all__ = ["DatasetProvider", "ensure_registered"]
