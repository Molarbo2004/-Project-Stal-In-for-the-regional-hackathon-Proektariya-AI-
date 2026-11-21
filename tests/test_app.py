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
        assert result.context["status"] == "Есть сдвиг"
        assert len(result.context["predictions"]) == 128
        exported = client.get("/batches/predictions.csv")
        assert exported.status_code == 200 and len(exported.text.splitlines()) == 129
        assert "/review/retrain" not in client.get("/review").text
        assert post("/review/retrain", result, batch=result.context["batch"].id).status_code == 409
        review = client.get("/review", params={"batch": result.context["batch"].id})
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
        assert [m["number"] for m in client.get("/api/models").json()] == [1, 0, 2, 3]
        assert client.post("/batches/demo", data={"csrf": "incorrect"}).status_code == 403
        assert client.get("/openapi.json").json()["paths"]["/api/predict"]

        records = load_uci(APP.parent)[0].iloc[:2].to_json(orient="records")
        files = {"file": ("small.json", records, "application/json")}
        optuna = post("/models/select", home, number=0, destination="quality")
        assert optuna.context["version"].number == 0 and "Optuna" in optuna.text
        optuna_result = client.post("/api/predict?version=0", files=files).json()
        assert optuna_result["version"] == 0 and len(optuna_result["predictions"]) == 2
        forecast = client.post("/api/predict?version=3", files=files).json()
        assert forecast["version"] == 3 and len(forecast["predictions"]) == 2
        assert forecast["monitoring"] == "insufficient_data"
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
        assert "Обнаружен сдвиг признаков" in client.get("/quality").text
        rejected = client.post("/batches/upload", data={"csrf": home.context["csrf"]},
                               files={"file": ("bad.json", "[{}]", "application/json")})
        assert rejected.status_code == 422 and rejected.context["issues"]
        assert client.get("/").context["batch"] is None

        experiment = client.get("/experiment?run=true&target=0.75&minutes=3&rubles=40")
        assert experiment.status_code == 200 and experiment.context["drift"].trials == 40
        assert "до дополнительной разметки" in experiment.text
        revised = client.get("/experiment?run=true&target=0.793&minutes=5&rubles=60")
        assert "/experiment/chart.png?target=0.793" in revised.text
        assert "<td>80</td><td>400</td><td>4800</td>" in revised.text
        assert client.get("/experiment/chart.png?target=0.75").content != client.get("/experiment/chart.png?target=0.793").content
        assert client.get("/experiment/chart.png").content.startswith(b"\x89PNG")
        assert len(client.get("/experiment/results.csv").text.splitlines()) == 31
    assert all(p.read_bytes() == contents for p, contents in artifacts.items())
