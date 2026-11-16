"""User-facing errors.

Raise :class:`SamotateError` for problems the user can fix (missing input,
bad config, nothing to do). The CLI prints these as one-line messages instead
of tracebacks.
"""


class SamotateError(Exception):
    """A problem with the user's input or project state, not a bug."""
