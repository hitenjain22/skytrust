"""Exceptions shared by training and inference code (kept dependency-free)."""


class LeakageError(AssertionError):
    """Train and test data overlap in time."""
