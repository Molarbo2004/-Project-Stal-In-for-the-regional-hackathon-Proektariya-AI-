from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, t
from sklearn.metrics import f1_score


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


RATING_CRITERIA = (
    ("recall", "Полнота по классам"),
    ("macro_f1", "Macro-F1"),
    ("accuracy", "Accuracy"),
    ("worst_fold", "Худшая accuracy 5×2"),
    ("speed", "Скорость обучения"),
)


def model_rating(evaluations, cv, folds):
    fastest = min(cv[name]["fit_seconds"] for name in evaluations)
    rating = {}
    for name, metrics in evaluations.items():
        components = {
            "recall": float(metrics.per_class.recall.mean()),
            "macro_f1": metrics.macro_f1,
            "accuracy": metrics.accuracy,
            "worst_fold": min(row[name] for row in folds),
            "speed": fastest / cv[name]["fit_seconds"],
        }
        rating[name] = components

    values = pd.DataFrame.from_dict(rating, orient="index")
    ranks = values.rank(ascending=False, method="average")
    mean_ranks = ranks.mean(axis=1).sort_values(kind="stable")
    return {
        name: {"components": rating[name], "ranks": ranks.loc[name].to_dict(), "mean_rank": mean_rank}
        for name, mean_rank in mean_ranks.items()
    }

def summarize_fivefold(oof, names):
    summary = {}
    for name in names:
        scores = []
        for _, fold in oof.groupby("fold"):
            truth, pred = fold.truth, fold[name.lower()]
            scores.append({"accuracy": (truth == pred).mean(),
                           "macro_f1": f1_score(truth, pred, average="macro"),
                           "weighted_f1": f1_score(truth, pred, average="weighted")})
        frame = pd.DataFrame(scores)
        summary[name] = {**frame.mean().to_dict(), "accuracy_std": frame.accuracy.std(ddof=0),
                         "macro_f1_std": frame.macro_f1.std(ddof=0)}
    return dict(sorted(summary.items(), key=lambda item: -item[1]["weighted_f1"]))
