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
from faults.model import split_dataset


def main():
    directory = ROOT / "base model"
    x, labels = load_uci(ROOT)
    train = x.index.difference(split_dataset(x, labels).test[0].index)
    classes = (ROOT / "Faults27x7_var").read_text().split()[27:]
    truth = pd.Categorical(labels.loc[train], categories=classes).codes
    rows = []
    for number, name in MODELS.items():
        artifact = directory / f"model_{name.lower()}.pkl"
        with artifact.open("rb") as stream:
            pipeline = _BaseModelUnpickler(str(artifact), stream, ensure_native_byte_order=True).load()
        recipe = clone(pipeline).set_params(clf__device="cpu", clf__n_jobs=2)
        features = BaseModel(directory, name, classes)._prepare_features(x.loc[train])
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
    pd.DataFrame(rows).to_csv(ROOT / "analysis" / "primary_5x2.csv", index=False)


if __name__ == "__main__":
    main()
