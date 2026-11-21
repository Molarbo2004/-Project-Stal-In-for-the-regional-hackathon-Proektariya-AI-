from dataclasses import dataclass

import numpy as np
import pandas as pd

from faults.drift import DriftBaseline, assess_drift
from faults.model import DatasetSplit, evaluate, fit_model
from faults.review import rank_for_review

BRIGHTNESS = ("Minimum_of_Luminosity", "Maximum_of_Luminosity", "Sum_of_Luminosity", "Luminosity_Index")
DEMO_BUDGETS = (0, 10, 20, 40, 80)
DEMO_SEEDS = (42, 47, 53)


@dataclass(frozen=True)
class DriftDemo:
    detection_rate: float
    false_alert_rate: float
    trials: int
    batch_size: int


@dataclass(frozen=True)
class Savings:
    labels: int
    minutes: float
    rubles: float
    active_budget: int
    random_budget: int


def simulate_brightness_shift(baseline: DriftBaseline, batch: pd.DataFrame) -> pd.DataFrame:
    result = batch.copy()
    for feature in BRIGHTNESS:
        if feature not in result:
            continue
        values = baseline.reference[:, baseline.columns.index(feature)]
        q1, q3 = np.quantile(values, [.25, .75])
        result[feature] = result[feature] + 1.4 * max(q3 - q1, np.std(values) / 2)
    return result


def run_drift_demo(baseline: DriftBaseline, features: pd.DataFrame, seed: int = 42) -> DriftDemo:
    candidates = [size for size in baseline.thresholds if size <= min(len(features), 60)]
    if not candidates:
        raise ValueError("Для эксперимента по сдвигу нужно минимум 30 записей.")
    size = max(candidates)
    rng = np.random.default_rng(seed)
    trials = 40
    normal_count, shifted_count = 0, 0
    for _ in range(trials):
        original = features.iloc[rng.choice(len(features), size, replace=False)]
        shifted = simulate_brightness_shift(baseline, original)
        normal_count += assess_drift(baseline, original).status == "shift"
        shifted_count += assess_drift(baseline, shifted).status == "shift"
    return DriftDemo(shifted_count / trials, normal_count / trials, trials, size)


def compare_review(split: DatasetSplit, budgets: tuple[int, ...], seeds: tuple[int, ...]) -> pd.DataFrame:
    x_train, y_train = split.train
    x_pool, y_pool = split.pool
    if not budgets or not seeds or tuple(sorted(set(budgets))) != budgets or not 0 <= budgets[0] <= budgets[-1] <= len(x_pool):
        raise ValueError("Укажите возрастающие бюджеты в пределах пула и хотя бы одно семя.")
    results = []
    for seed in seeds:
        initial = fit_model(x_train, y_train, seed)
        initial_score = evaluate(initial, *split.test).macro_f1
        random_order = np.random.default_rng(seed).permutation(len(x_pool))
        for strategy in ("active", "random"):
            model = initial
            selected = []
            remaining = list(range(len(x_pool)))
            score = initial_score
            for budget in budgets:
                take = budget - len(selected)
                if take:
                    if strategy == "active":
                        local = rank_for_review(model, x_pool.iloc[remaining], x_train, take)
                        selected.extend(remaining[i] for i in local)
                        used = set(local)
                        remaining = [index for i, index in enumerate(remaining) if i not in used]
                    else:
                        selected = random_order[:budget].tolist()
                    x = pd.concat([x_train, x_pool.iloc[selected]], ignore_index=True)
                    y = pd.concat([y_train, y_pool.iloc[selected]], ignore_index=True)
                    model = fit_model(x, y, seed)
                    score = evaluate(model, *split.test).macro_f1
                results.append((strategy, seed, len(selected), score))
    return pd.DataFrame(results, columns=["strategy", "seed", "budget", "macro_f1"])


def estimate_savings(curves: pd.DataFrame, target_f1: float, minutes: float, rubles: float) -> Savings | None:
    mean_scores = curves.groupby(["strategy", "budget"], as_index=False)["macro_f1"].mean()
    achieved = mean_scores[mean_scores["macro_f1"] >= target_f1].groupby("strategy")["budget"].min()
    if not {"active", "random"}.issubset(achieved.index):
        return None
    active, random = int(achieved["active"]), int(achieved["random"])
    difference = random - active
    return Savings(difference, difference * minutes, difference * rubles, active, random)
