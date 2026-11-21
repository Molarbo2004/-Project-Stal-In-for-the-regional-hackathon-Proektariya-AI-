import numpy as np
import pandas as pd

from faults.drift import assess_drift, fit_drift_baseline


def test_drift_and_small_batch():
    rng = np.random.default_rng(42)
    a, b, c = [pd.DataFrame(rng.normal(size=(300, 2)), columns=["brightness", "area"]) for _ in range(3)]
    baseline = fit_drift_baseline(a, b)
    assert assess_drift(baseline, c.iloc[:29]).status == "insufficient_data"
    assert assess_drift(baseline, c.iloc[:70]).status == "stable"
    shifted = c.copy()
    shifted["brightness"] += 10
    result = assess_drift(baseline, shifted)
    assert result.status == "shift"
    assert result.scores.index[0] == "brightness"
