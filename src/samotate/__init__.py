"""samotate — SAM-assisted auto-labeling for object detection.

A command-line pipeline that samples frames from a video or image folder,
labels them with Segment Anything (SAM) text prompts, and exports a
YOLO-format dataset. An optional REST API exposes the same operations.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
