import os
import secrets
import sqlite3
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from faults.base_model import BaseModel
from faults.charts import plot_learning_curves
from faults.data import BatchInputError, import_batch, load_uci
from faults.drift import MIN_BATCH_SIZE, assess_drift, fit_drift_baseline
from faults.experiments import (DEMO_BUDGETS, DEMO_SEEDS, compare_review, estimate_savings,
                                run_drift_demo, simulate_brightness_shift)
from faults.model import evaluate, predict, split_dataset
from faults.review import rank_for_review
from faults.workspace import ModelVersion, Workspace

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
STATUS = {"shift": "Есть сдвиг", "stable": "Без сдвига", "insufficient_data": "Малая партия"}
PAGES = {"recognition": "Распознавание", "review": "Проверка", "quality": "Качество",
         "experiment": "Эксперимент"}
PREDICTION_COLUMNS = {"class": "Прогноз", "score": "Вероятность", "margin": "Разрыв топ-2"}
TEMPLATES = Jinja2Templates(directory=WEB / "templates")
site = APIRouter(include_in_schema=False)
api = APIRouter()


class StalIn:
    def __init__(self, workspace_path: Path):
        features, self.labels = load_uci(ROOT)
        self.split = split_dataset(features, self.labels)
        self.baseline = fit_drift_baseline(self.split.train[0], self.split.validation[0])
        model = BaseModel(ROOT / "base model")
        self.initial = ModelVersion(1, model, evaluate(model, *self.split.test), 0)
        self.workspace = Workspace(workspace_path)
        self._experiment_lock = Lock()
        self._experiment = None

    def model_version(self, number: int) -> ModelVersion:
        if number == 1:
            return self.initial
        try:
            return self.workspace.load_version(number)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    def experiment(self):
        with self._experiment_lock:
            if self._experiment is None:
                curves = compare_review(self.split, DEMO_BUDGETS, DEMO_SEEDS)
                drift = run_drift_demo(self.baseline, self.split.pool[0])
                self._experiment = curves, drift
        return self._experiment


def service(request: Request) -> StalIn:
    return request.app.state.stal_in


def chosen_model(request: Request) -> ModelVersion:
    qc = service(request)
    number = request.session.get("model", 1)
    return qc.model_version(number)


def check_csrf(request: Request, csrf: Annotated[str, Form()]):
    expected = request.session.get("csrf", "")
    if not expected or not secrets.compare_digest(expected.encode(), csrf.encode()):
        raise HTTPException(403, "Обновите страницу и повторите действие.")


def render(request: Request, page: str, *, status_code=200, version=None, **context):
    qc = service(request)
    if "csrf" not in request.session:
        request.session["csrf"] = secrets.token_urlsafe(32)
    context.update(page=page, pages=PAGES, csrf=request.session["csrf"],
                   version=version or chosen_model(request),
                   versions=[1, *qc.workspace.version_numbers()],
                   notice=request.session.pop("notice", None),
                   experiment_open=request.query_params.get("run") == "true")
    return TEMPLATES.TemplateResponse(
        request, f"{page}.html", context=context, status_code=status_code,
        headers={"Cache-Control": "no-store"},
    )


def redirect(request: Request, page: str, **query) -> RedirectResponse:
    # A path-only Location lets the client retain its public host and scheme behind a proxy.
    path = request.url_for(page)
    if query:
        path = path.include_query_params(**query)
    return RedirectResponse(path.path + (f"?{path.query}" if path.query else ""), status_code=303)


def load_upload(qc: StalIn, upload: UploadFile):
    payload = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Размер файла превышает 200 МБ.")
    filename = (upload.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    return filename, import_batch(payload, Path(filename).suffix, qc.baseline.columns)


def counts(labels):
    return list(labels.value_counts().items())


@site.get("/", response_class=HTMLResponse)
def recognition(request: Request):
    qc = service(request)
    version = chosen_model(request)
    batch_id = request.session.get("batch")
    batch = next((item for item in qc.workspace.list_batches() if item.id == batch_id), None) if batch_id else None
    context = {}
    if batch:
        frame = predict(version.model, qc.workspace.load_batch(batch.id))
        context = dict(status=STATUS[batch.status], categories=counts(frame["class"]),
                       predictions=frame.reset_index(names="row").to_dict("records"))
    return render(request, "recognition", batch=batch, version=version, **context)


@site.post("/batches/upload", dependencies=[Depends(check_csrf)])
def upload_batch(request: Request, file: UploadFile):
    qc = service(request)
    try:
        filename, batch = load_upload(qc, file)
    except BatchInputError as error:
        request.session.pop("batch", None)
        return render(request, "recognition", status_code=422, batch=None, issues=error.issues)
    request.session["batch"] = qc.workspace.register_batch(batch, assess_drift(qc.baseline, batch), filename)
    request.session.pop("review_batch", None)
    request.session["quality_batch"] = request.session["batch"]
    return redirect(request, "recognition")


@site.post("/batches/demo", dependencies=[Depends(check_csrf)])
def demo(request: Request):
    qc = service(request)
    batch = simulate_brightness_shift(qc.baseline, qc.split.pool[0].iloc[:128])
    request.session["batch"] = qc.workspace.register_batch(
        batch, assess_drift(qc.baseline, batch), "Пример со сдвигом яркости",
    )
    request.session.pop("review_batch", None)
    request.session["quality_batch"] = request.session["batch"]
    return redirect(request, "recognition")


@site.get("/batches/predictions.csv")
def prediction_csv(request: Request):
    batch_id = request.session.get("batch")
    if not batch_id:
        raise HTTPException(404, "Загрузите партию.")
    try:
        batch = service(request).workspace.load_batch(batch_id)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    frame = predict(chosen_model(request).model, batch).rename(columns=PREDICTION_COLUMNS)
    return csv_response(frame.to_csv(index_label="Запись"), "stal-in-predictions.csv")


@site.get("/review", response_class=HTMLResponse)
def review(request: Request, batch: str | None = None, position: int | None = None):
    qc = service(request)
    workspace = qc.workspace
    version = chosen_model(request)
    batches = workspace.list_batches()
    if not batches:
        return render(request, "review", batches=[], version=version)
    priorities = {item.id: item for item in batches}
    selected = batch or request.session.get("review_batch")
    current = priorities.get(selected)
    if batch and current is None:
        raise HTTPException(404, "Партия отсутствует.")
    if current is None:
        return render(request, "review", batches=batches, statuses=STATUS, version=version)
    request.session["review_batch"] = current.id
    request.session["quality_batch"] = current.id
    features = workspace.load_batch(current.id)
    report = assess_drift(qc.baseline, features)
    confirmed = workspace.labels_for(current.id)
    remaining = [i for i in range(len(features)) if i not in confirmed]
    order = rank_for_review(version.model, features.iloc[remaining], qc.split.train[0], 10)
    positions = [remaining[i] for i in order]
    queue = []
    if positions:
        predictions = predict(version.model, features.iloc[positions]).to_dict("records")
        queue = [dict(item, position=i) for i, item in zip(positions, predictions)]
    entry = next((item for item in queue if item["position"] == position), queue[0] if queue else None)
    characteristics = list(features.iloc[entry["position"]].items()) if entry else []
    eligible = workspace.editable_labels(qc.split.train, qc.split.validation, qc.split.test, batch_id=current.id)
    return render(request, "review", version=version, batches=batches, batch_id=current.id,
                  statuses=STATUS, report=report, minimum_batch=MIN_BATCH_SIZE, queue=queue,
                  entry=entry, characteristics=characteristics, eligible=len(eligible), batch_name=current.name,
                  categories=sorted(version.model.classes_))


@site.post("/review/label", dependencies=[Depends(check_csrf)])
def save_label(request: Request, batch: Annotated[str, Form()], position: Annotated[int, Form(ge=0)],
               label: Annotated[str, Form()]):
    qc = service(request)
    if label not in set(qc.labels):
        raise HTTPException(422, "Неизвестная категория дефекта.")
    try:
        qc.workspace.save_label(batch, position, label)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    request.session["notice"] = ("success", f"Метка записи {position + 1:03} сохранена · {label}")
    return redirect(request, "review", batch=batch)


@site.post("/review/retrain", dependencies=[Depends(check_csrf)])
def retrain(request: Request, batch: Annotated[str, Form()]):
    qc = service(request)
    if batch != request.session.get("review_batch"):
        raise HTTPException(409, "Сначала откройте выбранную партию для дообучения.")
    try:
        version = qc.workspace.retrain(qc.split.train, qc.split.test, validation=qc.split.validation,
                                       protected_policy="skip", batch_id=batch)
    except ValueError as error:
        request.session["notice"] = ("error", str(error))
    except (RuntimeError, sqlite3.Error, OSError):
        request.session["notice"] = ("error", "Не удалось сохранить новую версию модели.")
    else:
        active = request.session.get("model", 1)
        request.session["notice"] = ("success", f"Версия v{version.number} сохранена. Активна v{active}.")
    return redirect(request, "review", batch=batch)


@site.post("/models/select", dependencies=[Depends(check_csrf)])
def select_model(request: Request, number: Annotated[int, Form(ge=1)], destination: Annotated[str, Form()],
                 run: Annotated[bool, Form()] = False):
    qc = service(request)
    if number != 1 and number not in qc.workspace.version_numbers():
        raise HTTPException(404, "Версия модели отсутствует.")
    request.session["model"] = number
    page = destination if destination in PAGES else "recognition"
    return redirect(request, page, **({"run": "true"} if page == "experiment" and run else {}))


@site.get("/quality", response_class=HTMLResponse)
def quality(request: Request, batch: str | None = None):
    qc = service(request)
    version = chosen_model(request)
    batches = qc.workspace.list_batches()
    selected = batch or request.session.get("quality_batch") or request.session.get("batch")
    current = next((item for item in batches if item.id == selected), None)
    if batch and batch != "uci" and current is None:
        raise HTTPException(404, "Партия отсутствует.")
    request.session["quality_batch"] = current.id if current else "uci"
    metrics = version.evaluation
    batch_counts = []
    report = None
    confidence = None
    if current:
        features = qc.workspace.load_batch(current.id)
        predictions = predict(version.model, features)
        batch_counts = counts(predictions["class"])
        confidence = predictions["score"].mean()
        report = assess_drift(qc.baseline, features)
        labels = qc.workspace.labels_for(current.id)
        positions = sorted(labels)
        metrics = evaluate(version.model, features.iloc[positions], [labels[i] for i in positions]) if labels else None
    return render(request, "quality", version=version, metrics=metrics, batches=batches, current=current,
                  report=report, confidence=confidence, minimum_batch=MIN_BATCH_SIZE,
                  actual_counts=counts(qc.labels), batch_counts=batch_counts, actual_size=len(qc.labels))


@site.get("/experiment", response_class=HTMLResponse)
def experiment(request: Request, run: bool = False, target: Annotated[float, Query(ge=0, le=1)] = .793,
               minutes: Annotated[float, Query(ge=0, allow_inf_nan=False)] = 3,
               rubles: Annotated[float, Query(ge=0, allow_inf_nan=False)] = 40):
    if not run:
        return render(request, "experiment", run=False)
    curves, drift = service(request).experiment()
    return render(request, "experiment", run=True, drift=drift, target=target, minutes=minutes,
                  rubles=rubles, budgets=DEMO_BUDGETS, savings=estimate_savings(curves, target, minutes, rubles))


def csv_response(csv: str, filename: str):
    return Response(csv.encode("utf-8-sig"), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@site.get("/experiment/results.csv")
def experiment_csv(request: Request):
    curves, _ = service(request).experiment()
    return csv_response(curves.to_csv(index=False), "review-experiment.csv")


@site.get("/experiment/chart.png")
def experiment_chart(request: Request, target: Annotated[float, Query(ge=0, le=1)] = .793):
    curves, _ = service(request).experiment()
    image = BytesIO()
    plot_learning_curves(curves, target).savefig(image, format="png")
    return Response(image.getvalue(), media_type="image/png")


@site.get("/dashboard/data/{name:path}")
def dashboard_data(name: str):
    if name not in {"Faults27x7_var", "Faults.NNA", "analysis/benchmark.json", "analysis/holdout_predictions.csv"}:
        raise HTTPException(404)
    return FileResponse(ROOT / name)


@api.get("/healthz", tags=["Служебное"])
def health():
    return {"status": "ok"}


@api.get("/api/models", tags=["Модели"])
def models(request: Request):
    return [{"number": n, "base": n == 1} for n in [1, *service(request).workspace.version_numbers()]]


@api.post("/api/predict", tags=["Распознавание"])
def predict_batch(request: Request, file: UploadFile, version: Annotated[int, Query(ge=1)] = 1,
                  include_features: bool = False):
    """Классифицирует CSV, JSON или BSON без записи партии в рабочее пространство."""
    qc = service(request)
    model = qc.model_version(version)
    try:
        _, batch = load_upload(qc, file)
    except BatchInputError as error:
        issues = [{"row": row, "field": field, "message": message} for row, field, message in error.issues]
        raise HTTPException(422, issues) from error
    report = assess_drift(qc.baseline, batch)
    result = predict(model.model, batch).reset_index(names="row")
    if include_features:
        result["features"] = batch.loc[result["row"]].to_dict("records")
    return {"version": version, "monitoring": report.status, "predictions": result.to_dict("records")}


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        default_db = ROOT / ".stal-in" / "work.db"
        legacy_db = ROOT / ".steel-qc" / "work.db"
        if legacy_db.is_file() and not default_db.exists():
            default_db = legacy_db
        path = Path(os.environ.get("STAL_IN_WORKSPACE") or os.environ.get("STEEL_QC_WORKSPACE") or default_db)
        application.state.stal_in = StalIn(path)
        yield

    application = FastAPI(title="Stal-In", description="Классификация дефектов стальных листов", lifespan=lifespan)
    application.add_middleware(SessionMiddleware, secret_key=os.environ.get("STAL_IN_SESSION_SECRET")
                               or os.environ.get("STEEL_QC_SESSION_SECRET")
                               or secrets.token_urlsafe(32), session_cookie="stal_in",
                               https_only=os.environ.get("RENDER") == "true")
    application.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
    application.include_router(site)
    application.include_router(api)
    return application


app = create_app()
