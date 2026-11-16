"""Command-line entry point: ``samotate <command>``.

Commands:
  init       scaffold a config.yaml you can edit
  pipeline   run auto-labeling and export (extract, detect, reconcile, export)
  status     show progress for each pipeline stage
  preview    draw boxes onto image copies so you can check the labels
  serve      start the REST API (no web UI)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from . import __version__
from .config import ProjectConfig
from .core.sam import SamNotAvailable
from .errors import SamotateError

logger = logging.getLogger("samotate")

EXAMPLE_CONFIG = """\
# samotate project configuration
# Relative paths are resolved against this file's directory.
name: my-project

# --- source: set exactly ONE of these ---
# video: ./input.mp4        # sample frames from a video
images_dir: ./images        # or annotate an existing folder of images

root: ./dataset             # output / working directory
fps: 1.0                    # frame sampling rate (video sources only)
# limit: 200                # cap number of images (useful for a pilot run)

# --- label classes: ids must be 0..N-1; 'prompts' drive SAM detection ---
classes:
  - id: 0
    name: object
    prompts:
      - text: "an object"
        threshold: 0.5

# --- how SAM detections are resolved into boxes ---
reconcile:
  strategy: per_prompt      # or 'hierarchical' for base/subclass tasks
  iou_threshold: 0.4
  nms_iou: 0.5

# --- export ---
train_fraction: 0.8
link_mode: copy             # 'copy' (Windows-safe) or 'symlink'
"""


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def cmd_init(args: argparse.Namespace) -> int:
    out = Path(args.output)
    if out.exists() and not args.force:
        logger.error("%s already exists (use --force to overwrite)", out)
        return 1
    out.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    logger.info("wrote %s — edit it, then run `samotate pipeline -c %s --stage all`",
                out, out)
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    from .core import detect, export, extract, reconcile

    cfg = ProjectConfig.from_yaml(args.config)
    stage = args.stage
    if stage in ("extract", "all"):
        extract.gather_source_images(cfg)
    if stage in ("detect", "all"):
        detect.detect_all(cfg, force=args.force)
    if stage in ("reconcile", "all"):
        reconcile.reconcile_all(cfg)
    if stage in ("export", "all"):
        result = export.export_dataset(cfg, every=args.every, offset=args.offset)
        logger.info("export: %s", result)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .core.review import project_status

    cfg = ProjectConfig.from_yaml(args.config)
    s = project_status(cfg)
    print(f"project      {cfg.name}")
    print(f"root         {cfg.root}")
    print(f"frames       {s['frames']}")
    print(f"detections   {s['detections']}")
    print(f"annotations  {s['annotations']}  ({s['reviewed']} reviewed, {s['empty']} empty)")
    print(f"labels       {s['labels']}")
    print("boxes")
    width = max(len(n) for n in s["boxes"])
    for name, n in s["boxes"].items():
        print(f"  {name:<{width}}  {n}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    from .core.review import render_previews

    cfg = ProjectConfig.from_yaml(args.config)
    out = render_previews(cfg, out_dir=args.output, limit=args.limit)
    print(out)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server.app import create_app

    app = create_app(config_path=args.config, workspace=args.workspace)
    logger.info("samotate API at http://%s:%d (docs at /docs)", args.host, args.port)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning("the API has no authentication; anyone who can reach "
                       "%s:%d can read and write your projects", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="samotate", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"samotate {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("init", help="scaffold a config.yaml")
    pi.add_argument("-o", "--output", default="config.yaml")
    pi.add_argument("--force", action="store_true")
    pi.set_defaults(func=cmd_init)

    pp = sub.add_parser("pipeline", help="run auto-labeling and export")
    pp.add_argument("-c", "--config", required=True)
    pp.add_argument("--stage", required=True,
                    choices=["extract", "detect", "reconcile", "export", "all"])
    pp.add_argument("--force", action="store_true",
                    help="re-run detect even if cached")
    pp.add_argument("--every", type=_positive_int, default=1,
                    help="export only every N-th frame")
    pp.add_argument("--offset", type=_non_negative_int, default=0,
                    help="first frame to use with --every")
    pp.set_defaults(func=cmd_pipeline)

    pst = sub.add_parser("status", help="show progress for each pipeline stage")
    pst.add_argument("-c", "--config", required=True)
    pst.set_defaults(func=cmd_status)

    ppv = sub.add_parser("preview", help="draw boxes onto image copies")
    ppv.add_argument("-c", "--config", required=True)
    ppv.add_argument("-o", "--output", default=None,
                     help="output directory (default: <root>/preview)")
    ppv.add_argument("-n", "--limit", type=_positive_int, default=None,
                     help="render only the first N images")
    ppv.set_defaults(func=cmd_preview)

    ps = sub.add_parser("serve", help="start the REST API (no web UI)")
    ps.add_argument("-c", "--config", default=None,
                    help="project config.yaml; its project's parent dir becomes the workspace")
    ps.add_argument("-w", "--workspace", default=None,
                    help="directory holding projects (default: current directory)")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)
    ps.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except ValidationError as exc:
        logger.error("invalid config:\n%s", exc)
        return 1
    except (SamotateError, SamNotAvailable, FileNotFoundError, ValueError) as exc:
        if args.verbose:
            logger.exception("failed")
        else:
            logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
