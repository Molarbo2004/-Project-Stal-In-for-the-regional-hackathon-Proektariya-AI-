import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


class BatchInputError(ValueError):
    def __init__(self, issues: list[tuple[int, str, str]]):
        self.issues = issues
        super().__init__(issues[0][2])


def load_uci(directory: Path) -> tuple[pd.DataFrame, pd.Series]:
    columns = (directory / "Faults27x7_var").read_text(encoding="utf-8-sig").split()
    if len(columns) != 34 or len(set(columns)) != 34:
        raise ValueError("Ожидалось 27 имён признаков и 7 имён классов UCI.")
    raw = np.loadtxt(directory / "Faults.NNA", ndmin=2)
    if raw.shape[1] != 34 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Файл Faults.NNA должен содержать 34 конечных числа на строку.")
    labels = raw[:, 27:]
    if not np.isin(labels, [0, 1]).all() or not (labels.sum(axis=1) == 1).all():
        raise ValueError("Для каждой записи UCI ожидается ровно одна положительная метка.")
    x = pd.DataFrame(raw[:, :27], columns=columns[:27])
    y = pd.Series(np.asarray(columns[27:])[labels.argmax(axis=1)], name="class", index=x.index)
    return x, y


class JsonRecord(dict):
    def __init__(self, pairs):
        super().__init__()
        self.duplicates = []
        for key, value in pairs:
            if key in self:
                self.duplicates.append(key)
            self[key] = value


def parse_csv(payload: bytes, features: tuple[str, ...]) -> list[dict]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
    columns = reader.fieldnames or []
    repeated = sorted({col for col in columns if columns.count(col) > 1})
    issues = [(0, col, "Повторяющийся заголовок.") for col in repeated]
    issues.extend((0, col, "Нет столбца в заголовке.") for col in features if col not in columns)
    if issues:
        raise BatchInputError(issues)
    return list(reader)


def parse_records(payload: bytes, file_format: str, features: tuple[str, ...]) -> list:
    if file_format == "csv":
        return parse_csv(payload, features)
    if file_format == "json":
        source = json.loads(payload.decode("utf-8-sig"), object_pairs_hook=JsonRecord, parse_int=float)
        return [source] if isinstance(source, dict) else source
    if file_format == "bson":
        from bson import BSON
        from bson.errors import InvalidBSON

        try:
            source = BSON(payload).decode()
        except InvalidBSON as error:
            raise BatchInputError([(0, "", "Некорректный документ BSON.")]) from error
        return source["records"] if "records" in source else [source]
    raise BatchInputError([(0, "", "Поддерживаются CSV, JSON и BSON.")])


def numeric_value(value, file_format: str) -> float:
    if file_format == "bson":
        from bson.decimal128 import Decimal128

        if isinstance(value, Decimal128):
            value = float(value.to_decimal())
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError
    if isinstance(value, str) and file_format != "csv":
        raise ValueError
    number = float(value)
    if not math.isfinite(number) or abs(number) > np.finfo(np.float32).max:
        raise ValueError
    return number


def import_batch(payload: bytes, file_format: str, features: tuple[str, ...]) -> pd.DataFrame:
    kind = file_format.casefold().lstrip(".")
    try:
        records = parse_records(payload, kind, features)
    except (UnicodeError, json.JSONDecodeError, csv.Error, RecursionError) as error:
        raise BatchInputError([(0, "", f"Не удалось прочитать файл {kind.upper()}.")]) from error
    if not isinstance(records, list) or not records:
        raise BatchInputError([(0, "", "Файл должен содержать хотя бы одну запись.")])
    issues = []
    values = []
    for row, record in enumerate(records, 1):
        if not isinstance(record, dict):
            issues.append((row, "", "Ожидалась запись с именованными полями."))
            continue
        if None in record:
            issues.append((row, "", "В строке больше значений, чем в заголовке CSV."))
        if isinstance(record, JsonRecord):
            issues.extend((row, key, "Повторяющееся поле JSON.") for key in record.duplicates)
        fields = []
        for name in features:
            if name not in record or record[name] is None:
                issues.append((row, name, "Отсутствует значение."))
                continue
            try:
                fields.append(numeric_value(record[name], kind))
            except (ValueError, OverflowError):
                issues.append((row, name, "Ожидалось конечное число в диапазоне модели."))
        values.append(fields)
    if issues:
        raise BatchInputError(issues)
    return pd.DataFrame(values, columns=list(features), index=range(1, len(values) + 1))
