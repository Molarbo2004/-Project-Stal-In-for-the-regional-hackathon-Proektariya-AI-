from pathlib import Path

import numpy as np
from joblib.numpy_pickle import NumpyUnpickler
from xgboost import XGBClassifier


class _SavedBooster:
    def __setstate__(self, state):
        # The snapshot contains GPU training configuration and a portable UBJSON model.
        # Load the model alone so the training configuration is never applied on CPU.
        snapshot = state["handle"]
        model_key = b"L" + (5).to_bytes(8, "big") + b"Model"
        start = snapshot.index(model_key) + len(model_key)
        self.model_bytes = snapshot[start:-1]


class _BaseModelUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if (module, name) == ("xgboost.core", "Booster"):
            return _SavedBooster
        return super().find_class(module, name)


MODELS = {0: "Baseline", 1: "GridSearch", 1.1: "Optuna"}


class BaseModel:
    def __init__(self, directory: Path, name: str, classes):
        self.classes_ = np.asarray(classes)
        artifact = directory / f"model_{name.lower()}.pkl"
        with artifact.open("rb") as file:
            pipeline = _BaseModelUnpickler(str(artifact), file, ensure_native_byte_order=True).load()
        self.features = list(pipeline.feature_names_in_)
        self.classifier = XGBClassifier(device="cpu", n_jobs=2)
        self.classifier.load_model(pipeline.named_steps["clf"].get_booster().model_bytes)

    def _prepare_features(self, batch):
        width = batch["X_Maximum"] - batch["X_Minimum"]
        height = batch["Y_Maximum"] - batch["Y_Minimum"]
        area = batch["Pixels_Areas"] + 1e-6
        return batch.assign(
            Aspect_Ratio=width / (height + 1e-6),
            Lum_per_Pixel=batch["Sum_of_Luminosity"] / area,
            Perim_per_Pixel=(batch["X_Perimeter"] + batch["Y_Perimeter"]) / area,
            XY_Size_Index=height / (width + 1e-6),
        ).loc[:, self.features]

    def predict(self, batch):
        return self.classes_[self.classifier.predict(self._prepare_features(batch))]

    def predict_proba(self, batch):
        return self.classifier.predict_proba(self._prepare_features(batch))
