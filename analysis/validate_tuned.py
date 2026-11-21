"""Compare the saved classifiers and their CPU training recipes on Steel Plates Faults."""
import sys
from time import perf_counter
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from faults.base_model import MODELS, BaseModel, _BaseModelUnpickler
from faults.data import load_uci
from faults.model import predict


def main():
    directory = ROOT / "base model"
    x, labels = load_uci(ROOT)
    oof = pd.DataFrame({"source_line": x.index + 1, "truth": labels})
    classes = (ROOT / "Faults27x7_var").read_text().split()[27:]
    truth = pd.Categorical(labels, categories=classes).codes
    rows = []
    for name in MODELS.values():
        artifact = directory / f"model_{name.lower()}.pkl"
        with artifact.open("rb") as stream:
            pipeline = _BaseModelUnpickler(str(artifact), stream, ensure_native_byte_order=True).load()
        recipe = clone(pipeline).set_params(clf__device="cpu", clf__n_jobs=2)
        features = BaseModel(directory, name, classes)._prepare_features(x)
        key = name.lower()
        parts = []
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(features, truth), 1):
            fitted = clone(recipe).fit(features.iloc[tr], truth[tr])
            part = predict(fitted, features.iloc[te])
            part["class"] = part["class"].map(dict(enumerate(classes)))
            parts.append(part)
            oof.loc[te, "fold"] = fold
        predictions = pd.concat(parts).sort_index()
        oof[key] = predictions["class"]
        oof[key + "_margin"] = predictions["margin"]
        oof[key + "_score"] = predictions["score"]
        for repeat, seed in enumerate((42, 47, 53, 59, 67), 1):
            folds = StratifiedKFold(2, shuffle=True, random_state=seed)
            for fold, (tr, te) in enumerate(folds.split(features, truth), 1):
                start = perf_counter()
                fitted = clone(recipe).fit(features.iloc[tr], truth[tr])
                elapsed = perf_counter() - start
                predicted = fitted.predict(features.iloc[te])
                rows.append(dict(model=name, repeat=repeat, fold=fold, fit_seconds=elapsed,
                                 accuracy=accuracy_score(truth[te], predicted),
                                 macro_f1=f1_score(truth[te], predicted, average="macro")))
    oof.astype({"fold": int}).to_csv(ROOT / "analysis" / "primary_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "analysis" / "primary_5x2.csv", index=False)


if __name__ == "__main__":
    main()
