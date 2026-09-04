"""Utilities for hydra parameter sweeps."""


def constant_list(length, value):
    """Create a list repeating a single value."""
    return [value] * length
