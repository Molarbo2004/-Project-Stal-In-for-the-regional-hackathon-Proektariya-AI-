from pathlib import Path

import pytest

from faults.data import load_uci
from faults.drift import DriftReport
from faults.model import split_dataset
from faults.workspace import Workspace


def test_persistence_priority_and_atomic_failure(tmp_path, monkeypatch):
    dataset = split_dataset(*load_uci(Path(__file__).resolve().parents[1]))
    workspace = Workspace(tmp_path / "work.db")
    x, y = dataset.pool
    normal = workspace.register_batch(x.iloc[:2], DriftReport("stable"))
    alert = workspace.register_batch(x.iloc[2:4], DriftReport("shift"))
    protected = workspace.register_batch(dataset.test[0].iloc[:1], DriftReport("insufficient_data"))
    assert workspace.register_batch(x.iloc[:2], DriftReport("stable")) == normal
    assert workspace.list_batches()[0].id == alert
    workspace.save_label(normal, 0, y.iloc[0])
    workspace.save_label(normal, 0, y.iloc[0])
    workspace.save_label(alert, 0, y.iloc[2])
    version = workspace.retrain(dataset.train, dataset.test, batch_id=normal)
    assert version.number == 2 and version.label_count == 1
    reopened = Workspace(tmp_path / "work.db")
    assert reopened.current_version().number == 2
    assert reopened.labels_for(normal) == {0: y.iloc[0]}
    reopened.save_label(protected, 0, dataset.test[1].iloc[0])
    with pytest.raises(ValueError, match="контрольн"):
        reopened.retrain(dataset.train, dataset.test, batch_id=protected)
    assert reopened.current_version().number == 2
    reopened.save_label(normal, 1, y.iloc[1])
    monkeypatch.setattr("faults.workspace.fit_model", lambda *_args: (_ for _ in ()).throw(RuntimeError("fit")))
    with pytest.raises(RuntimeError, match="fit"):
        reopened.retrain(dataset.train, dataset.test, protected_policy="skip", batch_id=normal)
    assert reopened.current_version().number == 2
    assert reopened.labels_for(normal) == {0: y.iloc[0], 1: y.iloc[1]}
