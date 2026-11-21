from pathlib import Path

from faults.data import load_uci
from faults.model import evaluate, fit_model, predict, split_dataset


def test_split_and_holdout():
    x, y = load_uci(Path(__file__).resolve().parents[1])
    a = split_dataset(x, y)
    b = split_dataset(x, y)
    parts = [a.train, a.validation, a.pool, a.test]
    sets = [set(part[0].index) for part in parts]
    assert len(set.union(*sets)) == sum(map(len, sets)) == len(x)
    assert a.test[0].index.equals(b.test[0].index)
    assert all(set(part[1]) == set(y) for part in parts)
    model = fit_model(*a.train)
    batch = a.test[0].iloc[:1]
    predictions = predict(model, batch)
    assert predictions.index.equals(batch.index)
    assert predictions.iloc[0]["class"] in set(y)
    scores = evaluate(model, *a.test)
    assert scores.size == len(a.test[0])
    assert scores.accuracy > 0.65
    assert scores.macro_f1 > 0.6
    assert set(scores.per_class.index) == set(y)
