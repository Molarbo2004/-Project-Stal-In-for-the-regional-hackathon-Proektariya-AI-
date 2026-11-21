import numpy as np
import pandas as pd


def rank_for_review(model, batch: pd.DataFrame, reference: pd.DataFrame, limit: int) -> list[int]:
    limit = min(max(0, limit), len(batch))
    if not limit:
        return []
    top_two = np.sort(model.predict_proba(batch), axis=1)[:, -2:]
    margin = top_two[:, 1] - top_two[:, 0]
    candidates = np.argsort(margin, kind="stable")[:max(limit * 4, 20)]
    reference = reference.loc[:, list(batch.columns)].to_numpy(dtype=float)
    q1, q3 = np.quantile(reference, [.25, .75], axis=0)
    scale = np.where(q3 > q1, q3 - q1, np.maximum(np.std(reference, axis=0), 1))
    data = (batch.iloc[candidates].to_numpy(dtype=float) - np.median(reference, axis=0)) / scale
    data = np.clip(data, -8, 8)

    def distance(position):
        return np.linalg.norm(data - data[position], axis=1) / np.sqrt(data.shape[1])

    selected = [0]
    nearest = distance(0)
    available = np.ones(len(candidates), dtype=bool)
    available[0] = False
    for _ in range(1, limit):
        priority = 1 - margin[candidates] + .35 * np.minimum(nearest, 3)
        position = int(np.argmax(np.where(available, priority, -np.inf)))
        available[position] = False
        selected.append(position)
        nearest = np.minimum(nearest, distance(position))
    return candidates[selected].tolist()
