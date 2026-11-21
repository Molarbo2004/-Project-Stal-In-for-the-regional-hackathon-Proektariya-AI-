"""Audit the disputed holdout rows and test a training-only exclusion rule."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits

from compare_models import ROOT, SEEDS, fit, load_uci, split_dataset

OTHER = "Other_Faults"
MODELS = ["xgb", "hist", "smote"]


def neighborhood(reference, query=None):
    transform = QuantileTransformer(n_quantiles=min(200, len(reference)), random_state=42)
    model = NearestNeighbors(n_neighbors=5).fit(transform.fit_transform(reference))
    # With X=None sklearn excludes the example itself from its neighborhood.
    return model.kneighbors(None if query is None else transform.transform(query), return_distance=False)


def main():
    x, y = load_uci(ROOT)
    train, labels = split_dataset(x, y).train
    predictions = pd.read_csv(ROOT / "analysis/holdout_predictions.csv")
    disputed = predictions.loc[(predictions.truth == OTHER) & (predictions.top3_errors == 3)].copy()
    sample = x.loc[disputed.source_line - 1]
    neighbors = neighborhood(train, sample)
    votes = labels.to_numpy()[neighbors]
    disputed["other_among_five"] = (votes == OTHER).sum(axis=1)
    disputed["bumps_among_five"] = (votes == "Bumps").sum(axis=1)
    disputed["neighbor_labels"] = [", ".join(row) for row in votes]
    disputed["neighbor_source_lines"] = [", ".join(map(str, train.index[row]+1)) for row in neighbors]
    for col in ("Pixels_Areas", "Steel_Plate_Thickness", "TypeOfSteel_A300"):
        disputed[col] = sample[col].to_numpy()
    disputed.to_csv(ROOT/"analysis/disputed_other_faults.csv", index=False, encoding="utf-8-sig")
    print("all duplicated:", x.duplicated(keep=False).sum())
    print("duplicates excluding positions:", sample.drop(columns=["X_Minimum","X_Maximum","Y_Minimum","Y_Maximum"]).duplicated().sum())
    print(disputed[["source_line","xgb","hist","smote","other_among_five","bumps_among_five","Pixels_Areas","Steel_Plate_Thickness","TypeOfSteel_A300"]].to_string(index=False))
    print("zero other", (disputed.other_among_five==0).sum(), "majority bumps", (disputed.bumps_among_five>=3).sum())
    for seed in SEEDS:
        estimates = [np.empty(len(labels),dtype=object) for _ in range(2)]
        removed = []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(train,labels):
            categories = labels.iloc[tr].to_numpy()
            neighbors = neighborhood(train.iloc[tr])
            exclude = (categories == OTHER) & ((categories[neighbors] == OTHER).sum(axis=1)==0)
            removed.append(int(exclude.sum()))
            for out, selected in zip(estimates,(tr,tr[~exclude])):
                model = fit("Random Forest",train.iloc[selected],labels.iloc[selected],seed)
                out[va] = model.predict(train.iloc[va])
        print("seed",seed,"excluded",removed)
        for out in estimates:
            print("macro_f1, accuracy, other recall:", f1_score(labels,out,average="macro"),
                  accuracy_score(labels,out), (out[labels==OTHER]==OTHER).mean())


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
