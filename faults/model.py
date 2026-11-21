from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

Dataset = tuple[pd.DataFrame, pd.Series]


@dataclass(frozen=True)
class DatasetSplit:
    train: Dataset
    validation: Dataset
    pool: Dataset
    test: Dataset


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    macro_f1: float
    per_class: pd.DataFrame
    size: int


def split_dataset(features: pd.DataFrame, labels: pd.Series, seed: int = 42) -> DatasetSplit:
    indices = np.arange(len(features))
    rest, test = train_test_split(indices, test_size=.20, stratify=labels, random_state=seed)
    train, other = train_test_split(rest, test_size=.375, stratify=labels.iloc[rest], random_state=seed)
    validation, pool = train_test_split(other, test_size=.50, stratify=labels.iloc[other], random_state=seed)

    def part(positions):
        return features.iloc[positions], labels.iloc[positions]

    return DatasetSplit(part(train), part(validation), part(pool), part(test))


def fit_model(features: pd.DataFrame, labels: pd.Series, seed: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=160, random_state=seed)
    return model.fit(features, labels)


def predict(model, features: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(features)
    winners = probabilities.argmax(axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    return pd.DataFrame({
        "class": model.classes_[winners],
        "score": sorted_probabilities[:, -1],
        "margin": sorted_probabilities[:, -1] - sorted_probabilities[:, -2],
    }, index=features.index)


def evaluate(model, features: pd.DataFrame, labels: pd.Series) -> Evaluation:
    predictions = model.predict(features)
    classes = sorted(set(model.classes_) | set(labels))
    p, r, f1, count = precision_recall_fscore_support(labels, predictions, labels=classes, zero_division=0)
    report = pd.DataFrame({"precision": p, "recall": r, "f1": f1, "support": count}, index=classes)
    return Evaluation(float(accuracy_score(labels, predictions)), float(f1.mean()), report, len(features))
