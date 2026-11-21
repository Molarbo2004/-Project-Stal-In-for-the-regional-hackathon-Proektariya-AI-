from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, t


def compare_predictions(results, truth, names):
    pairs = []
    for a, b in combinations(names, 2):
        left = results[a.lower()] == truth
        right = results[b.lower()] == truth
        wins, losses = int((left & ~right).sum()), int((right & ~left).sum())
        p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
        pairs.append((a, b, wins, losses, p))
    return pairs


def load_comparison(path: Path, names):
    folds = pd.read_csv(path)
    summary = folds.groupby("model").agg(
        accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"), fit_seconds=("fit_seconds", "sum")
    ).to_dict("index")
    accuracy = folds.pivot(index=["repeat", "fold"], columns="model", values="accuracy")
    pairs = []
    for a, b in combinations(names, 2):
        differences = (accuracy[a] - accuracy[b]).unstack()
        variance = (differences[1] - differences[2]).pow(2).mean() / 2
        statistic = differences.loc[1, 1] / variance**0.5 if variance > 0 else None
        p = 2 * t.sf(abs(statistic), df=5) if statistic is not None else None
        pairs.append((a, b, statistic, p))
    return summary, accuracy.reset_index().to_dict("records"), pairs
