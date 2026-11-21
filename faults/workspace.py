import hashlib
import json
import pickle
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from faults.drift import DriftReport
from faults.model import Dataset, Evaluation, evaluate, fit_model


@dataclass(frozen=True)
class ModelVersion:
    number: int | float
    model: object
    evaluation: Evaluation
    label_count: int


@dataclass(frozen=True)
class BatchSummary:
    id: str
    name: str
    status: str
    count: int


def row_signature(columns, values) -> str:
    # The model uses float32 internally. Protect observations that are identical at that precision.
    numbers = np.array(values, dtype="<f4", copy=True)
    numbers[numbers == 0] = 0
    return hashlib.sha256(json.dumps(list(columns)).encode() + numbers.tobytes()).hexdigest()


def encode_frame(frame: pd.DataFrame) -> str:
    # Python's JSON representation round-trips doubles without losing significant digits.
    return json.dumps({"columns": list(frame.columns), "data": frame.to_numpy().tolist()})


def decode_frame(payload: str) -> pd.DataFrame:
    values = json.loads(payload)
    return pd.DataFrame(values["data"], columns=values["columns"],
                        index=pd.RangeIndex(1, len(values["data"]) + 1), dtype=float)


class Workspace:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                    count INTEGER NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_rows (
                    batch_id TEXT REFERENCES batches(id), position INTEGER,
                    signature TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(batch_id, position)
                );
                CREATE TABLE IF NOT EXISTS labels (
                    signature TEXT PRIMARY KEY, label TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS versions (
                    number INTEGER PRIMARY KEY, label_count INTEGER NOT NULL,
                    feedback_hash TEXT NOT NULL, payload BLOB NOT NULL
                );
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @contextmanager
    def transaction(self):
        with closing(self.connect()) as db, db:
            yield db

    def register_batch(self, features: pd.DataFrame, drift_report: DriftReport, name: str = "Партия") -> str:
        payload = encode_frame(features)
        columns, values = list(features.columns), features.to_numpy().tolist()
        batch_id = hashlib.sha256(payload.encode()).hexdigest()
        with self.transaction() as db:
            insert = db.execute(
                "INSERT OR IGNORE INTO batches VALUES (?, ?, ?, ?, ?)",
                (batch_id, name, drift_report.status, len(values), payload),
            )
            if insert.rowcount:
                db.executemany("INSERT INTO batch_rows VALUES (?, ?, ?, ?)", (
                    (batch_id, position, row_signature(columns, row), json.dumps([columns, row]))
                    for position, row in enumerate(values)
                ))
        return batch_id

    def load_batch(self, batch_id: str) -> pd.DataFrame:
        with self.transaction() as db:
            row = db.execute("SELECT payload FROM batches WHERE id=?", (batch_id,)).fetchone()
        if row is None:
            raise ValueError("Партия отсутствует.")
        return decode_frame(row[0])

    def list_batches(self) -> list[BatchSummary]:
        with self.transaction() as db:
            rows = db.execute("""
                SELECT id, name, status, count FROM batches
                ORDER BY CASE status WHEN 'shift' THEN 0 ELSE 1 END, rowid DESC
            """).fetchall()
        return [BatchSummary(*row) for row in rows]

    def labels_for(self, batch_id: str) -> dict[int, str]:
        with self.transaction() as db:
            return dict(db.execute("""
                SELECT b.position, l.label FROM batch_rows AS b
                JOIN labels AS l USING(signature) WHERE b.batch_id=?
            """, (batch_id,)).fetchall())

    def save_label(self, batch_id: str, row: int, label: str) -> None:
        if not label:
            raise ValueError("Выберите категорию.")
        with self.transaction() as db:
            current = db.execute("""
                SELECT signature, payload FROM batch_rows WHERE batch_id=? AND position=?
            """, (batch_id, row)).fetchone()
            if current is None:
                raise ValueError("Запись не найдена.")
            signature, payload = current
            db.execute("""
                INSERT INTO labels VALUES (?, ?, ?)
                ON CONFLICT(signature) DO UPDATE SET label=excluded.label
            """, (signature, label, payload))

    @staticmethod
    def _batch_feedback(db, batch_id):
        return db.execute("""
            SELECT signature, label, payload FROM labels
            WHERE signature IN (SELECT signature FROM batch_rows WHERE batch_id=?)
            ORDER BY signature
        """, (batch_id,)).fetchall()

    def editable_labels(self, train: Dataset, *protected: Dataset, batch_id: str) -> list[tuple]:
        columns = list(train[0].columns)
        known = {
            row_signature(columns, values)
            for x, _ in (train, *protected)
            for values in x.to_numpy()
        }
        with self.transaction() as db:
            rows = self._batch_feedback(db, batch_id)
        return [row for row in rows if row[0] not in known]

    def current_version(self) -> ModelVersion | None:
        with self.transaction() as db:
            row = db.execute("SELECT payload FROM versions ORDER BY number DESC LIMIT 1").fetchone()
        # Only local app-generated model versions go into this workspace.
        return pickle.loads(row[0]) if row else None

    def version_numbers(self) -> list[int]:
        with self.transaction() as db:
            return [number for (number,) in db.execute("SELECT number FROM versions ORDER BY number")]

    def load_version(self, number: int) -> ModelVersion:
        with self.transaction() as db:
            row = db.execute("SELECT payload FROM versions WHERE number=?", (number,)).fetchone()
        if row is None:
            raise ValueError(f"Версия v{number} отсутствует.")
        return pickle.loads(row[0])

    def retrain(
        self, train: Dataset, test: Dataset, *, batch_id: str, validation: Dataset | None = None,
        protected_policy: str = "raise",
    ) -> ModelVersion:
        columns = list(train[0].columns)
        protected = (test, validation) if validation is not None else (test,)
        blocked = {
            row_signature(columns, values) for x, _ in protected for values in x.to_numpy()
        }
        with self.transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            feedback = self._batch_feedback(db, batch_id)
            if protected_policy == "raise" and any(row[0] in blocked for row in feedback):
                raise ValueError("Найдены контрольные записи, исключённые из дообучения.")
            existing = {row_signature(columns, x) for x in train[0].to_numpy()}
            allowed = [row for row in feedback if row[0] not in blocked and row[0] not in existing]
            if not allowed:
                raise ValueError("Для дообучения нужны новые метки вне исходной и контрольных выборок.")
            fingerprint = hashlib.sha256(json.dumps(allowed).encode()).hexdigest()
            last = db.execute("""
                SELECT number, feedback_hash, payload FROM versions ORDER BY number DESC LIMIT 1
            """).fetchone()
            if last and last[1] == fingerprint:
                return pickle.loads(last[2])
            values, labels = [], []
            classes = set(train[1])
            for _, label, payload in allowed:
                names, row = json.loads(payload)
                if names != columns or label not in classes:
                    raise ValueError("Разметка содержит неизвестную категорию или схему признаков.")
                values.append(row)
                labels.append(label)
            x = pd.concat([train[0], pd.DataFrame(values, columns=columns)], ignore_index=True)
            y = pd.concat([train[1], pd.Series(labels)], ignore_index=True)
            model = fit_model(x, y)
            version = ModelVersion(last[0] + 1 if last else 2, model, evaluate(model, *test), len(values))
            db.execute("INSERT INTO versions VALUES (?, ?, ?, ?)", (
                version.number, version.label_count, fingerprint, pickle.dumps(version),
            ))
        return version
