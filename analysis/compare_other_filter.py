"""Paired evaluation of the Other_Faults exclusion rule across model families."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from compare_models import MODELS, ROOT, SEEDS, fit, load_uci, split_dataset
from other_faults import OTHER, neighborhood


def main():
    features, labels = split_dataset(*load_uci(ROOT)).train
    x = features.to_numpy()
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    other = encoder.transform([OTHER])[0]
    rows = []
    for seed in SEEDS:
        partitions = []
        for train, valid in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
            neighbors = neighborhood(features.iloc[train])
            remove = (y[train] == other) & ~(y[train][neighbors] == other).any(axis=1)
            partitions.append((train, train[~remove], valid))
        for model in MODELS:
            for filtered in (False, True):
                pred = np.full(len(y), -1)
                for full, kept, valid in partitions:
                    train = kept if filtered else full
                    pred[valid] = fit(model, x[train], y[train], seed).predict(x[valid])
                assert np.all(pred >= 0)
                rows.append(dict(model=model, seed=seed, filtered=filtered,
                                 macro_f1=f1_score(y, pred, average="macro"),
                                 accuracy=accuracy_score(y, pred),
                                 other_recall=np.mean(pred[y == other] == other)))
            print(seed, model, flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "analysis/other_filter_results.csv", index=False)
    print(results.groupby(["model", "filtered"])[["macro_f1", "other_recall", "accuracy"]].mean().round(4))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
