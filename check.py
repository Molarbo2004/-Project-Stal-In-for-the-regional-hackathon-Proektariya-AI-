from pathlib import Path
import sklearn, numpy, xgboost, joblib, warnings
warnings.simplefilter("always")
print("sklearn", sklearn.__version__, "| numpy", numpy.__version__,
      "| xgboost", xgboost.__version__, "| joblib", joblib.__version__)

from faults.data import load_uci
from faults.model import split_dataset, evaluate
from faults.base_model import BaseModel
from faults.drift import fit_drift_baseline

ROOT = Path(__file__).resolve().parent
X, y = load_uci(ROOT)
print("UCI:", X.shape, "| классов:", y.nunique())

split = split_dataset(X, y)
print("split train/val/pool/test:",
      len(split.train[0]), len(split.validation[0]), len(split.pool[0]), len(split.test[0]))

model = BaseModel(ROOT / "base model")
print("модель загружена, классы:", list(model.classes_))
p = model.predict_proba(X.iloc[:5])
print("predict_proba:", p.shape, "| суммы:", p.sum(axis=1).round(3))

ev = evaluate(model, *split.test)
print("holdout:", ev.size, "| accuracy", round(ev.accuracy, 4), "| macro-F1", round(ev.macro_f1, 4))

base = fit_drift_baseline(split.train[0], split.validation[0])
print("пороги KS:", base.thresholds)
print("\nВСЁ ГОТОВО — можно запускать uvicorn")