import pandas as pd

from faults.experiments import compare_review, estimate_savings, run_drift_demo
from faults.data import load_uci
from faults.drift import fit_drift_baseline
from faults.model import split_dataset
from pathlib import Path


def test_savings_uses_observed_budgets_and_signed_difference():
    curves = pd.DataFrame({
        "strategy": ["active", "active", "random", "random"],
        "budget": [10, 20, 10, 20],
        "macro_f1": [.65, .8, .75, .8],
    })
    savings = estimate_savings(curves, .75, 3, 40)
    assert (savings.labels, savings.minutes, savings.rubles) == (-10, -30, -400)
    assert estimate_savings(curves, .9, 3, 40) is None


def test_experiments_are_paired_and_repeatable():
    split = split_dataset(*load_uci(Path(__file__).resolve().parents[1]))
    a = compare_review(split, (0, 3), (42,))
    b = compare_review(split, (0, 3), (42,))
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("budget")["strategy"].nunique().to_dict() == {0: 2, 3: 2}
    assert a.loc[a.budget == 0, "macro_f1"].nunique() == 1
    baseline = fit_drift_baseline(split.train[0], split.validation[0])
    result = run_drift_demo(baseline, split.pool[0])
    assert 0 <= result.false_alert_rate <= result.detection_rate <= 1
