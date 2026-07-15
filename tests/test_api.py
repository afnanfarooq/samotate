"""API tests with FastAPI's TestClient and a mock SAM backend."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from samotate.server.app import create_app

from .conftest import MockBackend


@pytest.fixture
def client(tmp_path):
    # workspace holds projects; images live in a separate source folder
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    images_dir = tmp_path / "src_images"
    images_dir.mkdir()
    for i in range(2):
        Image.fromarray(np.zeros((80, 60, 3), "uint8")).save(images_dir / f"f{i}.png")

    app = create_app(workspace=str(workspace))
    # inject the mock backend so assist works without a model
    app.state.samotate._backend = MockBackend()
    c = TestClient(app)
    c._images_dir = str(images_dir)
    return c


def _create(client):
    resp = client.post(
        "/api/projects",
        json={
            "name": "Demo",
            "images_dir": client._images_dir,
            "classes": [{"id": 0, "name": "thing", "prompts": [{"text": "thing"}]}],
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_create_list_extract_images(client):
    pid = _create(client)
    assert any(p["id"] == pid for p in client.get("/api/projects").json())

    assert client.post(f"/api/projects/{pid}/extract").json()["frames"] == 2
    images = client.get(f"/api/projects/{pid}/images").json()
    assert len(images) == 2
    assert images[0]["reviewed"] is False

    # serve raw bytes
    name = images[0]["name"]
    assert client.get(f"/api/projects/{pid}/images/{name}").status_code == 200


def test_annotation_roundtrip(client):
    pid = _create(client)
    client.post(f"/api/projects/{pid}/extract")
    name = client.get(f"/api/projects/{pid}/images").json()[0]["name"]

    # empty by default
    ann = client.get(f"/api/projects/{pid}/images/{name}/annotation").json()
    assert ann["boxes"] == []
    assert ann["width"] == 60 and ann["height"] == 80

    # save a box, read it back
    ann["boxes"] = [{"class_id": 0, "xyxy": [1, 2, 3, 4], "source": "manual"}]
    put = client.put(f"/api/projects/{pid}/images/{name}/annotation", json=ann)
    assert put.json()["saved"] is True

    reread = client.get(f"/api/projects/{pid}/images/{name}/annotation").json()
    assert reread["reviewed"] is True
    assert len(reread["boxes"]) == 1


def test_detect_interactive(client):
    pid = _create(client)
    client.post(f"/api/projects/{pid}/extract")
    name = client.get(f"/api/projects/{pid}/images").json()[0]["name"]

    resp = client.post(
        f"/api/projects/{pid}/detect-interactive",
        json={"image": name, "prompt": "thing", "threshold": 0.5},
    )
    assert resp.status_code == 200
    assert len(resp.json()["boxes"]) == 1


def test_export(client):
    pid = _create(client)
    client.post(f"/api/projects/{pid}/extract")
    client.post(f"/api/projects/{pid}/detect")
    # reconcile happens via pipeline; here just export whatever annotations exist
    name = client.get(f"/api/projects/{pid}/images").json()[0]["name"]
    ann = client.get(f"/api/projects/{pid}/images/{name}/annotation").json()
    ann["boxes"] = [{"class_id": 0, "xyxy": [1, 2, 30, 40], "source": "manual"}]
    client.put(f"/api/projects/{pid}/images/{name}/annotation", json=ann)

    result = client.post(f"/api/projects/{pid}/export", json={}).json()
    assert "data_yaml" in result
