# samotate

**Open-source, SAM-assisted image annotation tool with YOLO dataset export.**

samotate is a local, browser-based annotator for object detection. Draw and
correct bounding boxes by hand, or let [Segment Anything (SAM)](#sam-setup-optional)
pre-label your images from plain-text prompts — then export a ready-to-train
YOLO dataset. It works for **any** classes you define, not a fixed task.

> samotate began as an internal script that auto-labeled retail "staff vs.
> customer" footage. It's now a generic tool: classes and prompts come from a
> config file, and the original pipeline is the headless engine underneath the UI.

---

## Features

- 🖱️ **Interactive web annotator** — draw/move/resize/delete boxes on a canvas, assign classes by keyboard.
- 🧠 **SAM assist (optional)** — type a text prompt or click a point to get box suggestions you can accept.
- 🏷️ **Fully generic** — arbitrary classes and prompts via a YAML config.
- 🎞️ **Video or images** — sample frames from a video or annotate an existing folder.
- 📦 **YOLO export** — labels, a temporal train/val split, and `data.yaml`, ready for Ultralytics YOLO.
- ⚙️ **Headless pipeline** — auto-label and export from the command line, no UI.

## Install

```bash
pip install samotate
```

This gives you the annotator and manual labeling. SAM auto-labeling is an
optional extra (see [SAM setup](#sam-setup-optional)).

## Quickstart

```bash
samotate init                 # writes a config.yaml to edit
samotate serve                # opens the annotator in your browser
```

Or point at a specific project config:

```bash
samotate serve -c path/to/config.yaml
```

In the UI: create a project (from a video or image folder), define your classes,
optionally run auto-detect, then review/correct boxes and **Export**.

### Headless pipeline

The original automated workflow is still available end-to-end:

```bash
samotate pipeline -c config.yaml --stage all
# or one stage at a time: extract | detect | reconcile | export
```

## Configuration

A minimal `config.yaml`:

```yaml
name: my-project
images_dir: ./images          # or: video: ./input.mp4
root: ./dataset
classes:
  - id: 0
    name: cat
    prompts:
      - { text: "a cat", threshold: 0.5 }
  - id: 1
    name: dog
    prompts:
      - { text: "a dog", threshold: 0.5 }
reconcile:
  strategy: per_prompt        # 'hierarchical' for base/subclass tasks
train_fraction: 0.8
link_mode: copy               # 'copy' (Windows-safe) or 'symlink'
```

See [`examples/config.example.yaml`](examples/config.example.yaml) for the full
reference, including the `hierarchical` reconcile strategy (a broad base class
subdivided by specific classes, with override classes) used by the original
staff/customer pipeline.

## SAM setup (optional)

Manual annotation needs nothing extra. For text-prompt / click auto-labeling you
need a SAM backend. SAM 3.1 is a research package and is **not on PyPI** — install
PyTorch and the model per its own instructions, then run samotate on a machine
with a GPU (CUDA is detected automatically; CPU/MPS also work but are slow):

```bash
pip install "samotate[sam]"   # installs torch
# then install SAM 3.1 from its source per upstream docs
```

If no backend is available, samotate runs in manual-only mode and assist calls
return a clear "SAM not installed" message instead of failing silently.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup and how to build the
frontend into the package.

## License

[MIT](LICENSE)
