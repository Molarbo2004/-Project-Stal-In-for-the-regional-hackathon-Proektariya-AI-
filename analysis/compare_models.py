"""Reproduce the Steel QC benchmark from the project directory."""
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from faults.data import load_uci
from faults.model import split_dataset

SEEDS = (42, 47, 53)
MODELS = ("Logistic regression", "Random Forest", "Gradient boosting (hist)", "XGBoost",
          "XGBoost + SMOTE", "Extra Trees")


def smote(x, y, seed):
    """Standard k=5 SMOTE in standardized feature space; balance each class to majority size."""
    rng = np.random.default_rng(seed)
    scaled = StandardScaler().fit_transform(x)
    target = np.bincount(y).max()
    features, labels = [x], [y]
    for label, count in enumerate(np.bincount(y)):
        if count == target:
            continue
        points = x[y == label]
        neighbors = NearestNeighbors(n_neighbors=5).fit(scaled[y == label]).kneighbors(return_distance=False)
        starts = rng.integers(count, size=target - count)
        ends = neighbors[starts, rng.integers(5, size=len(starts))]
        fractions = rng.random((len(starts), 1))
        features.append(points[starts] + fractions * (points[ends] - points[starts]))
        labels.append(np.full(len(starts), label))
    return np.vstack(features), np.concatenate(labels)


def fit(name, x, y, seed):
    if name == "Logistic regression":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1))
    elif name == "Random Forest":
        model = RandomForestClassifier(n_estimators=160, random_state=seed)
    elif name == "Gradient boosting (hist)":
        model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
                                               learning_rate=.07, l2_regularization=1, random_state=seed)
    elif name == "Extra Trees":
        model = ExtraTreesClassifier(n_estimators=250, random_state=seed)
    else:
        if name == "XGBoost + SMOTE":
            x, y = smote(x, y, seed)
        model = XGBClassifier(n_estimators=180, max_depth=4, learning_rate=.06, subsample=.9,
                              colsample_bytree=.9, tree_method="hist", n_jobs=2, random_state=seed)
    return model.fit(x, y)


def main():
    split = split_dataset(*load_uci(ROOT))
    encoder = LabelEncoder().fit(split.train[1])
    x, y = split.train[0].to_numpy(), encoder.transform(split.train[1])
    folds, repetitions = [], []
    for seed in SEEDS:
        partitions = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y))
        for name in MODELS:
            oof = np.empty_like(y)
            for fold, (train, valid) in enumerate(partitions, 1):
                start = perf_counter()
                model = fit(name, x[train], y[train], seed)
                seconds = perf_counter() - start
                pred = model.predict(x[valid])
                oof[valid] = pred
                folds.append(dict(model=name, seed=seed, fold=fold, fit_seconds=seconds,
                                  macro_f1=f1_score(y[valid], pred, average="macro"),
                                  accuracy=accuracy_score(y[valid], pred)))
            repetitions.append(dict(model=name, seed=seed, macro_f1=f1_score(y, oof, average="macro"),
                                    accuracy=accuracy_score(y, oof)))
            print(seed, name, round(repetitions[-1]["macro_f1"], 4), flush=True)
    frame = pd.DataFrame(repetitions)
    summary = frame.groupby("model")[["macro_f1", "accuracy"]].mean().sort_values("macro_f1", ascending=False)
    winner = summary.index[0]
    holdout = {}
    for name in dict.fromkeys([winner, "Random Forest"]):
        model = fit(name, x, y, 42)
        pred = encoder.inverse_transform(model.predict(split.test[0].to_numpy()).astype(int))
        holdout[name] = classification_report(split.test[1], pred, output_dict=True, zero_division=0)
    result = dict(versions={"sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
                  train_count=len(y), holdout_count=len(split.test[0]), winner=winner,
                  summary=summary.to_dict(orient="index"), repetitions=repetitions, folds=folds, holdout=holdout)
    Path(__file__).with_name("benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
