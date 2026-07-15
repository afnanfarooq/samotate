"""Command-line entry point: ``samotate <command>``.

Commands:
  init       scaffold a config.yaml you can edit
  serve      start the FastAPI server and open the annotator in a browser
  pipeline   run the headless auto-label/export pipeline (the original workflow)
"""
from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

from . import __version__
from .config import ProjectConfig

logger = logging.getLogger("samotate")

EXAMPLE_CONFIG = """\
# samotate project configuration
name: my-project

# --- source: set exactly ONE of these ---
# video: ./input.mp4        # sample frames from a video
images_dir: ./images        # or annotate an existing folder of images

root: ./dataset             # output / working directory
fps: 1.0                    # frame sampling rate (video sources only)
# limit: 200                # cap number of images (useful for a pilot run)

# --- label classes; 'prompts' are optional SAM auto-label hints ---
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


def cmd_init(args: argparse.Namespace) -> int:
    out = Path(args.output)
    if out.exists() and not args.force:
        logger.error("%s already exists (use --force to overwrite)", out)
        return 1
    out.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    logger.info("wrote %s — edit it, then run `samotate serve`", out)
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


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server.app import create_app

    app = create_app(config_path=args.config)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        # open after a short delay so the server is up
        import threading

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    logger.info("serving samotate at %s", url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="samotate", description=__doc__)
    p.add_argument("--version", action="version", version=f"samotate {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("init", help="scaffold a config.yaml")
    pi.add_argument("-o", "--output", default="config.yaml")
    pi.add_argument("--force", action="store_true")
    pi.set_defaults(func=cmd_init)

    ps = sub.add_parser("serve", help="start the annotator web app")
    ps.add_argument("-c", "--config", default=None,
                    help="optional config.yaml to open on startup")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--no-browser", action="store_true")
    ps.set_defaults(func=cmd_serve)

    pp = sub.add_parser("pipeline", help="run the headless auto-label pipeline")
    pp.add_argument("-c", "--config", required=True)
    pp.add_argument("--stage", required=True,
                    choices=["extract", "detect", "reconcile", "export", "all"])
    pp.add_argument("--force", action="store_true",
                    help="re-run detect even if cached")
    pp.add_argument("--every", type=int, default=1)
    pp.add_argument("--offset", type=int, default=0)
    pp.set_defaults(func=cmd_pipeline)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
