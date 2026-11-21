from pathlib import Path

import pandas as pd
from scipy.stats import t


def load_comparison(directory: Path):
    holdout = pd.read_csv(directory / "tuned_optuna_paired.csv", index_col=0).loc["refit_holdout"]
    folds = pd.read_csv(directory / "tuned_optuna_5x2.csv").sort_values(["repeat", "fold"])
    # Within each repeat the two folds swap training and test halves.
    differences = folds.pivot(index="repeat", columns="fold", values="delta")
    variance = (differences[1] - differences[2]).pow(2).mean() / 2
    statistic = differences.loc[1, 1] / variance**0.5 if variance > 0 else None
    return {
        "holdout": holdout, "folds": folds.to_dict("records"),
        "tuned": folds.tuned.mean(), "optuna": folds.optuna.mean(),
        "statistic": statistic, "p": 2 * t.sf(abs(statistic), df=5) if statistic is not None else None,
    }
