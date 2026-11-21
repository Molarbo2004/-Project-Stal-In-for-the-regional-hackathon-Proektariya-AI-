from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app import create_app

from faults.charts import plot_class_counts, plot_drift_scores
from faults.data import load_uci

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_charts_render_without_a_window():
    counts = pd.Series({"Other_Faults": 42, "Bumps": 20})
    scores = pd.Series({"Minimum_of_Luminosity": .4, "Pixels_Areas": .2})
    for figure in (plot_class_counts(counts, "Частоты"), plot_drift_scores(scores)):
        buffer = BytesIO()
        figure.savefig(buffer, format="png")
        assert buffer.getvalue().startswith(b"\x89PNG")


def test_operator_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("STAL_IN_WORKSPACE", str(tmp_path / "work.db"))
    artifacts = {p: p.read_bytes() for p in (APP.parent / "base model").glob("*.pkl")}
    with TestClient(create_app()) as client:
        def post(path, page, **data):
            return client.post(path, data={"csrf": page.context["csrf"], **data})

        home = client.get("/")
        assert home.status_code == 200
        assert home.context["version"].number == 1
        assert client.get("/healthz").json() == {"status": "ok"}
        result = post("/batches/demo", home)
        workspace = client.app.state.stal_in.workspace
        sample = workspace.load_batch(result.context["batch"].id)
        source, _ = load_uci(APP.parent)
        assert sample.columns.equals(source.columns)
        assert pd.MultiIndex.from_frame(sample).isin(pd.MultiIndex.from_frame(source)).all()
        assert len(result.context["predictions"]) == 128
        exported = client.get("/batches/predictions.csv")
        assert exported.status_code == 200 and len(exported.text.splitlines()) == 129
        assert "/review/retrain" not in client.get("/review").text
        assert post("/review/retrain", result, batch=result.context["batch"].id).status_code == 409
        review = client.get("/review", params={"batch": result.context["batch"].id})
        assert review.context["report"].status == result.context["batch"].status
        for number in (2, 3):
            label = review.context["queue"][0]
            saved = post("/review/label", review, batch=review.context["batch_id"], position=label["position"],
                         label=label["class"])
            assert "сохранена" in saved.text
            trained = post("/review/retrain", saved, batch=saved.context["batch_id"])
            assert f"Версия v{number} сохранена" in trained.text
            assert trained.context["version"].number == 1
            quality = client.get("/quality")
            selected = post("/models/select", quality, number=number, destination="quality")
            assert selected.status_code == 404
            review = client.get("/review")
        assert client.get("/").context["version"].number == 1
        assert post("/models/select", review, number=1, destination="recognition").context["version"].number == 1
        client.cookies.clear()
        home = client.get("/")
        assert home.context["version"].number == 1
        assert [m["number"] for m in client.get("/api/models").json()] == [0, 1, 1.1]
        assert client.post("/batches/demo", data={"csrf": "incorrect"}).status_code == 403
        assert client.get("/openapi.json").json()["paths"]["/api/predict"]

        records = load_uci(APP.parent)[0].iloc[:2].to_json(orient="records")
        files = {"file": ("small.json", records, "application/json")}
        holdout = client.get("/dashboard/data/holdout.json").json()
        assert len(holdout) == 389
        for number, name in [(0, "Baseline"), (1, "GridSearch"), (1.1, "Optuna")]:
            quality = post("/models/select", home, number=number, destination="quality")
            assert quality.context["version"].number == number
            forecast = client.post(f"/api/predict?version={number}", files=files).json()
            assert forecast["version"] == number and len(forecast["predictions"]) == 2
            assert forecast["monitoring"] == "insufficient_data"
            key = name.lower()
            correct = sum(row[key] == row["truth"] for row in holdout)
            assert correct / len(holdout) == quality.context["version"].evaluation.accuracy
            assert all(0 <= row[key + "_margin"] <= row[key + "_score"] <= 1 for row in holdout)
        assert client.post("/api/predict?version=3", files=files).status_code == 404
        assert client.get("/").context["batch"] is None
        uploaded = client.post("/batches/upload", data={"csrf": home.context["csrf"]}, files=files)
        assert uploaded.context["batch"].name == "small.json"
        assert 'action="/batches/demo"' in uploaded.text
        batch_id = uploaded.context["batch"].id
        assert f'href="/quality?batch={batch_id}">Показать качество' in uploaded.text
        quality = client.get(f"/quality?batch={batch_id}")
        assert quality.context["metrics"] is None
        assert "Средняя уверенность модели" in quality.text
        assert "минимум 30 записей" in quality.text
        review = client.get("/review", params={"batch": batch_id})
        entry = review.context["queue"][0]
        post("/review/label", review, batch=batch_id, position=entry["position"], label=entry["class"])
        metrics = client.get("/quality").context["metrics"]
        assert metrics.size == 1 and metrics.accuracy == 1
        assert client.get("/quality?batch=uci").context["metrics"].size == 389
        assert post("/batches/demo", uploaded).context["batch"].count == 128
        assert client.get("/quality").context["current"].id == result.context["batch"].id
        rejected = client.post("/batches/upload", data={"csrf": home.context["csrf"]},
                               files={"file": ("bad.json", "[{}]", "application/json")})
        assert rejected.status_code == 422 and rejected.context["issues"]
        assert client.get("/").context["batch"] is None
    assert all(p.read_bytes() == contents for p, contents in artifacts.items())
