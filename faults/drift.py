from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_BATCH_SIZE = 30


@dataclass(frozen=True)
class DriftBaseline:
    columns: tuple[str, ...]
    reference: np.ndarray
    thresholds: dict[int, float]
    seed: int


@dataclass(frozen=True)
class DriftReport:
    status: str
    score: float | None = None
    threshold: float | None = None
    sample_size: int = 0
    scores: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def ks_scores(sorted_reference: np.ndarray, sample: np.ndarray) -> np.ndarray:
    scores = []
    for reference, values in zip(sorted_reference.T, np.sort(sample, axis=0).T):
        points = np.concatenate((reference, values))
        a = np.searchsorted(reference, points, side="right") / len(reference)
        b = np.searchsorted(values, points, side="right") / len(values)
        scores.append(np.max(np.abs(a - b)))
    return np.asarray(scores)


def fit_drift_baseline(
    reference: pd.DataFrame, validation: pd.DataFrame, seed: int = 42,
) -> DriftBaseline:
    if len(reference) < MIN_BATCH_SIZE or len(validation) < MIN_BATCH_SIZE:
        raise ValueError("Для настройки мониторинга требуется минимум 30 записей в каждой выборке.")
    columns = tuple(reference.columns)
    benchmark = np.sort(reference.to_numpy(dtype=float), axis=0)
    calibration = validation.loc[:, list(columns)].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    thresholds = {}
    for size in (MIN_BATCH_SIZE, 60, 100, 150, 200):
        if size > len(calibration):
            break
        maxima = [
            ks_scores(benchmark, calibration[rng.choice(len(calibration), size, replace=False)]).max()
            for _ in range(96)
        ]
        thresholds[size] = float(np.quantile(maxima, .95, method="higher"))
    return DriftBaseline(columns, benchmark, thresholds, seed)


def assess_drift(baseline: DriftBaseline, batch: pd.DataFrame) -> DriftReport:
    if len(batch) < MIN_BATCH_SIZE:
        return DriftReport("insufficient_data", sample_size=len(batch))
    size = max(size for size in baseline.thresholds if size <= len(batch))
    values = batch.loc[:, list(baseline.columns)].to_numpy(dtype=float)
    if len(batch) > size:
        values = values[np.random.default_rng(baseline.seed).choice(len(values), size, replace=False)]
    scores = pd.Series(ks_scores(baseline.reference, values), index=list(baseline.columns))
    scores = scores.sort_values(ascending=False, kind="stable")
    score, threshold = float(scores.iloc[0]), baseline.thresholds[size]
    return DriftReport("shift" if score > threshold else "stable", score, threshold, size, scores)
