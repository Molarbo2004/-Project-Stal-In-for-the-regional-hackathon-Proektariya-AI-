import numpy as np
import pandas as pd

from faults.review import rank_for_review


class Model:
    def predict_proba(self, batch):
        return np.array([[.5, .5], [.51, .49], [.52, .48], [.9, .1]])[:len(batch)]


def test_ranked_positions_are_unique_and_diverse():
    batch = pd.DataFrame({"shape": [0, 0, 5, 2]})
    assert rank_for_review(Model(), batch, batch, 2) == [0, 2]
    assert len(set(rank_for_review(Model(), batch, batch, 90))) == len(batch)
