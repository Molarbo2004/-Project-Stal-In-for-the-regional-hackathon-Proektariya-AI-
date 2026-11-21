import csv
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from faults.data import BatchInputError, import_batch, load_uci


def test_local_uci():
    features, labels = load_uci(Path(__file__).resolve().parents[1])
    assert features.shape == (1941, 27)
    assert labels.nunique() == 7
    assert labels.value_counts()["Other_Faults"] == 673


def test_formats_have_same_order():
    from bson import BSON

    features = ("x", "y")
    records = [{"y": 2, "x": 1}, {"x": 3, "y": 4}]
    payloads = {
        "csv": b"y,x\n2,1\n4,3\n",
        "json": json.dumps(records).encode(),
        "bson": BSON.encode({"records": records}),
    }
    expected = pd.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]}, index=[1, 2])
    for kind, payload in payloads.items():
        pd.testing.assert_frame_equal(import_batch(payload, kind, features), expected)
    assert len(import_batch(BSON.encode(records[0]), "bson", features)) == 1
    assert len(import_batch(json.dumps(records[0]).encode(), "json", features)) == 1


@pytest.mark.parametrize("payload,kind,field", [
    (b'[{"x":1,"y":2},{"x":true,"y":2}]', "json", "x"),
    (b'[{"x":1,"y":2},{"x":2}]', "json", "y"),
    (b"x,y\n1,2\ninf,2\n", "csv", "x"),
    (b"x,y\n1,2\n3,\n", "csv", "y"),
])
def test_invalid_value_identifies_record_and_field(payload, kind, field):
    with pytest.raises(BatchInputError) as error:
        import_batch(payload, kind, ("x", "y"))
    assert error.value.issues[0][:2] == (2, field)


def test_csv_duplicate_headers():
    with pytest.raises(BatchInputError) as error:
        import_batch(b"x,x\n1,2\n", "csv", ("x",))
    assert error.value.issues[0][:2] == (0, "x")


@pytest.mark.parametrize("payload", [
    b'[{"x":2},{"x":' + b"9" * 4301 + b"}]",
    b'[{"x":2},{"x":3,"x":4}]',
])
def test_json_error_in_second_record(payload):
    with pytest.raises(BatchInputError) as error:
        import_batch(payload, "json", ("x",))
    assert error.value.issues[0][:2] == (2, "x")
