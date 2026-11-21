"""Compare the saved classifiers and their CPU training recipes on Steel Plates Faults."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t, ttest_rel
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from faults.base_model import BaseModel, _BaseModelUnpickler
from faults.data import load_uci
from faults.model import split_dataset


def paired_scores(y, tuned, optuna):
    a, b = tuned == y, optuna == y
    wins, losses = int(sum(a & ~b)), int(sum(~a & b))
    test = ttest_rel(a.astype(float), b.astype(float))
    return dict(n=len(y), tuned_accuracy=a.mean(), optuna_accuracy=b.mean(),
                tuned_macro_f1=f1_score(y, tuned, average="macro"),
                optuna_macro_f1=f1_score(y, optuna, average="macro"),
                tuned_only=wins, optuna_only=losses,
                mcnemar_p=binomtest(wins, wins+losses).pvalue if wins+losses else 1.,
                paired_t=test.statistic, paired_p=test.pvalue)


def risk_scenarios(probs, truth, model):
    winner = probs.argmax(axis=1)
    risks = []
    for cost in [2,5,10]:
        threshold = 1 - 1/cost
        accept = probs.max(axis=1) >= threshold
        errors = int(np.sum(accept & (winner != truth)))
        review = int(sum(~accept))
        risks.append(dict(model=model, error_cost=cost, threshold=threshold, auto=int(sum(accept)), review=review,
                          auto_errors=errors, auto_accuracy=1-errors/sum(accept) if any(accept) else np.nan,
                          total_cost=review+cost*errors))
    return risks


def main():
    directory = ROOT / "base model"
    x, labels = load_uci(ROOT)
    holdout = split_dataset(x, labels).test[0].index.to_numpy()
    train = np.setdiff1d(np.arange(len(x)), holdout)
    saved, recipes = [], []
    for optuna, filename in [(False,"best_model_tuned.pkl"), (True,"best_model_optuna.pkl")]:
        saved.append(BaseModel(directory, optuna=optuna))
        with (directory / filename).open("rb") as stream:
            pipeline = _BaseModelUnpickler(str(directory / filename), stream, ensure_native_byte_order=True).load()
        recipes.append(clone(pipeline).set_params(clf__device="cpu", clf__n_jobs=2))
    assert np.array_equal(saved[0].classes_, saved[1].classes_)
    assert list(saved[0].features) == list(saved[1].features)
    features = saved[0]._prepare_features(x)
    y = pd.Categorical(labels, categories=saved[0].classes_).codes
    assert (y >= 0).all()
    predictions = [model.predict(x.iloc[holdout]) for model in saved]
    summary = {"saved_reference": paired_scores(labels.to_numpy()[holdout], *predictions)}
    fitted = [clone(model).fit(features.iloc[train], y[train]) for model in recipes]
    probs = fitted[0].predict_proba(features.iloc[holdout])
    truth = y[holdout]
    winner = probs.argmax(axis=1)
    summary["refit_holdout"] = paired_scores(truth, winner, fitted[1].predict(features.iloc[holdout]))
    risks = risk_scenarios(probs, truth, "refit_holdout")
    risks += risk_scenarios(saved[0].predict_proba(x.iloc[holdout]), truth, "saved_reference")
    folds = []
    for repeat, seed in enumerate((42,47,53,59,67),1):
        for fold,(tr,te) in enumerate(StratifiedKFold(2,shuffle=True,random_state=seed).split(features.iloc[train],y[train]),1):
            scores = [accuracy_score(y[train[te]], clone(model).fit(features.iloc[train[tr]],y[train[tr]]).predict(features.iloc[train[te]])) for model in recipes]
            folds.append(dict(repeat=repeat,fold=fold,tuned=scores[0],optuna=scores[1],delta=scores[0]-scores[1]))
    out = Path(__file__).resolve().parent
    frame = pd.DataFrame(folds)
    ds = frame.delta.to_numpy().reshape(5,2)
    variance = ((ds-ds.mean(axis=1,keepdims=True))**2).sum()/5
    statistic = ds[0,0]/np.sqrt(variance)
    print("5x2cv",frame[["tuned","optuna"]].mean().to_dict(),statistic,2*t.sf(abs(statistic),5))
    for name,values in summary.items(): print(name,values)
    print(pd.DataFrame(risks).to_string(index=False))
    frame.to_csv(out/"tuned_optuna_5x2.csv",index=False)
    pd.DataFrame(summary).T.to_csv(out/"tuned_optuna_paired.csv")
    pd.DataFrame(risks).to_csv(out/"tuned_risk_thresholds.csv",index=False)


if __name__ == "__main__":
    main()
