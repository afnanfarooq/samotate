"""CLI smoke tests: commands run headless and fail with one-line errors."""
import numpy as np
from PIL import Image

from samotate import cli
from samotate.core import detect


def _project(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for i in range(3):
        Image.fromarray(np.zeros((40, 60, 3), "uint8")).save(images / f"i{i}.png")
    cfg = tmp_path / "config.yaml"
    assert cli.main(["init", "-o", str(cfg)]) == 0
    return cfg


def test_full_pipeline_status_and_preview(tmp_path, monkeypatch, capsys, mock_backend):
    cfg = _project(tmp_path)
    monkeypatch.setattr(detect, "load_backend", lambda **_: mock_backend)
    monkeypatch.chdir("/")  # relative paths must resolve against the config

    assert cli.main(["pipeline", "-c", str(cfg), "--stage", "all"]) == 0
    assert (tmp_path / "dataset" / "data.yaml").exists()

    assert cli.main(["status", "-c", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "annotations  3" in out
    assert "object  3" in out

    assert cli.main(["preview", "-c", str(cfg)]) == 0
    assert len(list((tmp_path / "dataset" / "preview").glob("*.jpg"))) == 3


def test_errors_are_one_line_not_tracebacks(tmp_path, caplog):
    cfg = _project(tmp_path)
    assert cli.main(["pipeline", "-c", str(cfg), "--stage", "reconcile"]) == 1
    assert "run the 'detect' stage first" in caplog.text
    assert "Traceback" not in caplog.text


def test_invalid_config_reported(tmp_path, caplog):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("images_dir: ./x\nclasses: []\n")
    assert cli.main(["status", "-c", str(cfg)]) == 1
    assert "invalid config" in caplog.text
