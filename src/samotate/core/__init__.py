"""Core engine: frame extraction, SAM detection, reconciliation, YOLO export.

This package is a generalized refactor of the original ``build_dataset`` script.
Each module corresponds to one of the original pipeline stages, but is
config-driven and importable rather than hardcoded.
"""
