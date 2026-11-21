"""Run from the project root using the same environment as compare_models.py."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from compare_models import ROOT, SEEDS, fit, load_uci, split_dataset


def add_features(x):
    return x.assign(width=x.X_Maximum-x.X_Minimum, height=x.Y_Maximum-x.Y_Minimum,
                    brightness_range=x.Maximum_of_Luminosity-x.Minimum_of_Luminosity,
                    mean_brightness=x.Sum_of_Luminosity/x.Pixels_Areas.clip(lower=1))


def main():
    split = split_dataset(*load_uci(ROOT))
    train, labels = split.train
    test, truth = split.test
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    rows = pd.DataFrame({"source_line": test.index+1, "truth": truth.to_numpy()})
    models = {"xgb": "XGBoost", "hist": "Gradient boosting (hist)",
              "smote": "XGBoost + SMOTE", "rf": "Random Forest"}
    for key, name in models.items():
        model = fit(name, train.to_numpy(), y, 42)
        prob = model.predict_proba(test.to_numpy())
        rows[key] = encoder.inverse_transform(prob.argmax(axis=1))
        sorted_prob = np.sort(prob, axis=1)
        rows[key+"_margin"] = sorted_prob[:, -1] - sorted_prob[:, -2]
    wrong = rows[["xgb", "hist", "smote"]].ne(rows.truth, axis=0)
    rows["top3_errors"] = wrong.sum(axis=1)
    rows.sort_values(["top3_errors", "source_line"], ascending=[False, True]).to_csv(
        ROOT/"analysis"/"holdout_predictions.csv", index=False, encoding="utf-8-sig")
    for label, group in rows.groupby("truth"):
        print(label, len(group), *[int((group[key]!=label).sum()) for key in models],
              "all", (group.top3_errors==3).sum())
    for key in models:
        errors = rows[rows.truth != rows[key]]
        print(key, errors.groupby(["truth",key]).size().nlargest(5).to_dict())
    common = rows[rows.top3_errors==3]
    same = common[common.xgb.eq(common["hist"]) & common["hist"].eq(common.smote)]
    print("same", len(same), same.groupby(["truth","xgb"]).size().nlargest(6).to_dict())
    print("examples", same[["source_line","truth","xgb"]].head(5).to_dict("records"))
    all_features = add_features(train)
    for seed in SEEDS:
        base = np.empty_like(y)
        richer = np.empty_like(y)
        flagged = np.zeros(len(y), dtype=bool)
        for fit_idx, valid in StratifiedKFold(5, shuffle=True, random_state=seed).split(train,y):
            for data, predictions in [(train,base), (all_features,richer)]:
                model = fit("Random Forest", data.iloc[fit_idx].to_numpy(), y[fit_idx], seed)
                probs = model.predict_proba(data.iloc[valid].to_numpy())
                predictions[valid] = probs.argmax(axis=1)
                if data is train:
                    scores = np.sort(probs, axis=1)
                    order = np.argsort(scores[:,-1]-scores[:,-2],kind="stable")
                    flagged[valid[order[:round(.2*len(valid))]]] = True
        print(seed, "F1 base/extended", *(round(f1_score(y,p,average="macro"),4) for p in [base,richer]),
              "mistakes total/flagged; reviewed", int((base!=y).sum()), int(((base!=y)&flagged).sum()), flagged.sum())


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
