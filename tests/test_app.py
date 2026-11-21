from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app import create_app

from faults.charts import plot_class_counts, plot_drift_scores, plot_learning_curves
from faults.data import load_uci

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_charts_render_without_a_window():
    counts = pd.Series({"Other_Faults": 42, "Bumps": 20})
    scores = pd.Series({"Minimum_of_Luminosity": .4, "Pixels_Areas": .2})
    curves = pd.DataFrame({
        "strategy": ["active", "random"], "budget": [20, 20], "macro_f1": [.79, .77],
    })
    for figure in (plot_class_counts(counts, "Частоты"), plot_drift_scores(scores), plot_learning_curves(curves)):
        buffer = BytesIO()
        figure.savefig(buffer, format="png")
        assert buffer.getvalue().startswith(b"\x89PNG")


def test_operator_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("STEEL_QC_WORKSPACE", str(tmp_path / "work.db"))
    artifacts = {p: p.read_bytes() for p in (APP.parent / "base model").glob("*.pkl")}
    with TestClient(create_app()) as client:
        def post(path, page, **data):
            return client.post(path, data={"csrf": page.context["csrf"], **data})

        home = client.get("/")
        assert home.status_code == 200
        assert home.context["version"].number == 1
        assert client.get("/healthz").json() == {"status": "ok"}
        result = post("/batches/demo", home)
        assert result.context["status"] == "Есть сдвиг"
        assert len(result.context["predictions"]) == 128
        exported = client.get("/batches/predictions.csv")
        assert exported.status_code == 200 and len(exported.text.splitlines()) == 129
        review = client.get("/review")
        assert review.context["report"].status == "shift"
        for number in (2, 3):
            label = review.context["queue"][0]
            saved = post("/review/label", review, batch=review.context["batch_id"], position=label["position"],
                         label=label["class"])
            assert "сохранена" in saved.text
            trained = post("/review/retrain", saved, batch=saved.context["batch_id"])
            assert f"Версия v{number} сохранена" in trained.text
            assert trained.context["version"].number == number - 1
            quality = client.get("/quality")
            selected = post("/models/select", quality, number=number, destination="quality")
            assert selected.context["version"].number == number
            assert f"v{number}" in selected.text
            review = client.get("/review")
        assert client.get("/").context["version"].number == 3
        assert post("/models/select", review, number=1, destination="recognition").context["version"].number == 1
        client.cookies.clear()
        home = client.get("/")
        assert home.context["version"].number == 1
        assert [m["number"] for m in client.get("/api/models").json()] == [1, 2, 3]
        assert client.post("/batches/demo", data={"csrf": "incorrect"}).status_code == 403
        assert client.get("/openapi.json").json()["paths"]["/api/predict"]

        records = load_uci(APP.parent)[0].iloc[:2].to_json(orient="records")
        files = {"file": ("small.json", records, "application/json")}
        forecast = client.post("/api/predict?version=3", files=files).json()
        assert forecast["version"] == 3 and len(forecast["predictions"]) == 2
        assert forecast["monitoring"] == "insufficient_data"
        assert client.get("/").context["batch"] is None
        uploaded = client.post("/batches/upload", data={"csrf": home.context["csrf"]}, files=files)
        assert uploaded.context["batch"].name == "small.json"
        rejected = client.post("/batches/upload", data={"csrf": home.context["csrf"]},
                               files={"file": ("bad.json", "[{}]", "application/json")})
        assert rejected.status_code == 422 and rejected.context["issues"]
        assert client.get("/").context["batch"] is None

        experiment = client.get("/experiment?run=true&target=0.75&minutes=3&rubles=40")
        assert experiment.status_code == 200 and experiment.context["drift"].trials == 40
        assert client.get("/experiment/chart.png").content.startswith(b"\x89PNG")
        assert len(client.get("/experiment/results.csv").text.splitlines()) == 31
    assert all(p.read_bytes() == contents for p, contents in artifacts.items())
