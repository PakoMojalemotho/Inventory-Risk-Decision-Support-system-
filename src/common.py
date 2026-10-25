"""Shared data handling, model loading, scoring, and reporting helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"


def normalise_name(name: Any) -> str:
    """Convert a source column name to stable snake_case."""
    text = str(name).replace("\ufeff", "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize source columns and add deterministic BOP features used in BOTH train and score."""
    result = df.copy()
    result.columns = [normalise_name(c) for c in result.columns]
    result = result.loc[:, ~pd.Index(result.columns).duplicated()]
    # Keep engineered features deterministic and based only on predictor columns.
    def num(name: str) -> pd.Series | None:
        return pd.to_numeric(result[name], errors="coerce") if name in result.columns else None
    f3, f6, f9 = num("forecast_3_month"), num("forecast_6_month"), num("forecast_9_month")
    s1, s3, s6, s9 = num("sales_1_month"), num("sales_3_month"), num("sales_6_month"), num("sales_9_month")
    inv, transit, local_bo, min_bank = num("national_inv"), num("in_transit_qty"), num("local_bo_qty"), num("min_bank")
    if all(x is not None for x in (f3, f6, f9)):
        result["eng_total_forecast_3_6_9"] = f3.fillna(0) + f6.fillna(0) + f9.fillna(0)
        result["eng_forecast_monthly_peak"] = pd.concat([f3 / 3, f6 / 6, f9 / 9], axis=1).max(axis=1)
    sales_rates = []
    for series, months in ((s1, 1), (s3, 3), (s6, 6), (s9, 9)):
        if series is not None:
            sales_rates.append(series / months)
    if sales_rates:
        rates = pd.concat(sales_rates, axis=1)
        result["eng_monthly_sales_peak"] = rates.max(axis=1)
        result["eng_monthly_sales_mean"] = rates.mean(axis=1)
    if s1 is not None and s3 is not None:
        result["eng_recent_sales_change"] = s1 - (s3 / 3)
    if inv is not None:
        result["eng_inventory_position"] = inv.fillna(0) + (transit.fillna(0) if transit is not None else 0) - (local_bo.fillna(0) if local_bo is not None else 0)
        if min_bank is not None:
            result["eng_inventory_vs_min_bank"] = inv / min_bank.replace(0, np.nan)
    return result


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found: {source}\nPlace your cleaned Training BOP CSV in data/ "
            "or pass the actual path with --data."
        )
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return normalise_dataframe(pd.read_csv(source, encoding=encoding, low_memory=False))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode CSV {source}: {last_error}")


def detect_target(columns: list[str] | pd.Index, requested: str | None = None) -> str:
    available = {normalise_name(c) for c in columns}
    if requested:
        candidate = normalise_name(requested)
        if candidate not in available:
            raise ValueError(f"Requested target '{candidate}' was not found. Available columns: {sorted(available)}")
        return candidate
    # The known target of Training_BOP is went_on_backorder. Keep it first in the priority list.
    for candidate in (
        "went_on_backorder", "backorder", "backordered", "is_backordered",
        "backorder_flag", "target", "label", "class", "outcome"
    ):
        if candidate in available:
            return candidate
    raise ValueError(
        "Could not find a label column. Training BOP normally uses 'went_on_backorder'. "
        "Pass --target COLUMN if your cleaned file renamed it."
    )


def parse_binary_target(series: pd.Series) -> pd.Series:
    """Map common Yes/No, True/False, and 0/1 encodings to integer labels."""
    raw = series.copy()
    numeric = pd.to_numeric(raw, errors="coerce")
    nonmissing = raw.notna()
    unique_numeric = set(numeric[nonmissing & numeric.notna()].unique().tolist())
    if nonmissing.any() and numeric[nonmissing].notna().all() and unique_numeric.issubset({0, 1}):
        return numeric.astype("Int64")

    token_map = {
        "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1, "1.0": 1,
        "backorder": 1, "backordered": 1, "went_on_backorder": 1,
        "positive": 1, "no": 0, "n": 0, "false": 0, "f": 0,
        "0": 0, "0.0": 0, "not_backordered": 0, "no_backorder": 0,
        "negative": 0
    }
    normalized = raw.astype("string").str.strip().str.lower()
    mapped = normalized.map(token_map)
    if mapped[raw.notna()].notna().all():
        return mapped.astype("Int64")

    distinct = list(normalized.dropna().unique())
    if len(distinct) == 2:
        # Conservative fallback: infer the positive class only if a positive-looking label exists.
        positive_candidates = [v for v in distinct if any(t in str(v) for t in ("backorder", "positive", "yes", "true"))]
        if len(positive_candidates) == 1:
            positive = positive_candidates[0]
            return (normalized == positive).astype("Int64")
    raise ValueError(
        "Target must contain two recognisable classes (e.g. Yes/No or 1/0). "
        f"Observed values include: {distinct[:10]}"
    )


def is_identifier_column(column: str) -> bool:
    col = normalise_name(column)
    if col in {"sku", "id", "index", "row_id", "record_id", "item_id", "product_id", "part_id", "serial_number"}:
        return True
    return col.endswith("_id") or col in {"validation", "split", "dataset_split", "fold"}


def infer_feature_types(frame: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    boolean_like = {"yes", "no", "y", "n", "true", "false", "t", "f", "0", "1"}
    for col in columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
            numeric_cols.append(col)
            continue
        values = series.dropna().astype(str).str.strip().str.lower()
        if len(values) == 0:
            # Empty columns are treated as numerical and safely median-imputed later.
            numeric_cols.append(col)
            continue
        if set(values.unique()).issubset(boolean_like):
            categorical_cols.append(col)
            continue
        converted = pd.to_numeric(series, errors="coerce")
        parse_ratio = float(converted.notna().sum()) / max(1, int(series.notna().sum()))
        if parse_ratio >= 0.90:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def cast_feature_types(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    data = frame.copy()
    for col in numeric_columns:
        if col not in data:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
    for col in categorical_columns:
        if col not in data:
            data[col] = np.nan
        # Keep nulls as nulls so SimpleImputer can handle them.
        data[col] = data[col].astype("object")
        data.loc[data[col].notna(), col] = data.loc[data[col].notna(), col].astype(str).str.strip()
        data[col] = data[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return data[numeric_columns + categorical_columns]


def load_model_bundle(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run: python src/train.py --data data/Training_BOP.csv"
        )
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("The model file is not a compatible inventory BOP model bundle.")
    return bundle


def probability_to_risk(probability: float, medium_threshold: float = 0.25, high_threshold: float = 0.60) -> str:
    if probability >= high_threshold:
        return "High"
    if probability >= medium_threshold:
        return "Medium"
    return "Low"


def _first_numeric(row: pd.Series, candidates: tuple[str, ...]) -> float | None:
    for key in candidates:
        if key in row.index:
            value = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
            if pd.notna(value) and np.isfinite(value):
                return float(value)
    return None


def estimate_monthly_demand(row: pd.Series) -> float | None:
    """Conservative monthly demand proxy from BOP sales/forecast fields, not a trained forecast."""
    monthly_demand_candidates: list[float] = []
    sales1 = _first_numeric(row, ("sales_1_month", "sales_last_month"))
    sales3 = _first_numeric(row, ("sales_3_month", "sales_last_3_months"))
    sales6 = _first_numeric(row, ("sales_6_month", "sales_last_6_months"))
    sales9 = _first_numeric(row, ("sales_9_month", "sales_last_9_months"))
    forecast3 = _first_numeric(row, ("forecast_3_month", "forecast_next_3_months"))
    forecast6 = _first_numeric(row, ("forecast_6_month", "forecast_next_6_months"))
    forecast9 = _first_numeric(row, ("forecast_9_month", "forecast_next_9_months"))
    for value, divisor in ((sales1, 1), (sales3, 3), (sales6, 6), (sales9, 9),
                           (forecast3, 3), (forecast6, 6), (forecast9, 9)):
        if value is not None and value >= 0:
            monthly_demand_candidates.append(value / divisor)
    if not monthly_demand_candidates:
        return None
    monthly_demand = max(monthly_demand_candidates)
    return float(monthly_demand) if monthly_demand > 0 else None


def estimate_stock_cover_days(row: pd.Series) -> float | None:
    """Estimate days of cover from stock and BOP demand proxy; heuristic, not a demand forecast."""
    stock = _first_numeric(row, ("national_inv", "current_inventory", "inventory", "on_hand", "stock_on_hand"))
    if stock is None:
        return None
    transit = _first_numeric(row, ("in_transit_qty", "in_transit_qtry", "quantity_in_transit")) or 0.0
    monthly_demand = estimate_monthly_demand(row)
    if monthly_demand is None:
        return None
    return float((stock + max(transit, 0.0)) / monthly_demand * 30.0)


def make_recommendation(row: pd.Series, probability: float, medium_threshold: float = 0.25, high_threshold: float = 0.60) -> str:
    risk = probability_to_risk(probability, medium_threshold, high_threshold)
    stock = _first_numeric(row, ("national_inv", "current_inventory", "inventory", "on_hand", "stock_on_hand"))
    min_bank = _first_numeric(row, ("min_bank", "minimum_stock", "safety_stock"))
    lead_time = _first_numeric(row, ("lead_time", "supplier_lead_time"))
    cover = estimate_stock_cover_days(row)
    notes: list[str] = []
    if stock is not None and min_bank is not None and stock <= min_bank:
        notes.append("on-hand inventory is at or below the minimum-stock indicator")
    if cover is not None and lead_time is not None and cover <= lead_time:
        notes.append("estimated stock cover does not exceed supplier lead time")
    if risk == "High":
        action = "HIGH PRIORITY: verify physical stock and supplier status; review replenishment or expedited supply now."
    elif risk == "Medium":
        action = "MONITOR CLOSELY: check demand, available stock, and supplier lead time; prepare a replenishment decision if cover is insufficient."
    else:
        action = "ROUTINE MONITORING: keep stock and demand records current; avoid ordering excess stock based on the score alone."
    if notes:
        action += " Additional checks: " + "; ".join(notes) + "."
    action += " Human review is required before any purchase or expedite decision."
    return action


def score_records(
    records: pd.DataFrame,
    bundle: dict[str, Any],
    medium_threshold: float = 0.25,
    high_threshold: float = 0.60,
) -> pd.DataFrame:
    """Score a CSV/DataFrame while preserving its input columns for dashboard display."""
    original = records.copy()
    normalised = normalise_dataframe(records)
    feature_columns = bundle["feature_columns"]
    numeric_columns = bundle["numeric_columns"]
    categorical_columns = bundle["categorical_columns"]
    # Ignore any label accidentally included in a scoring upload.
    scoring_input = normalised.drop(columns=[bundle.get("target_column", "")], errors="ignore")
    x = cast_feature_types(scoring_input.reindex(columns=feature_columns), numeric_columns, categorical_columns)
    probabilities = bundle["pipeline"].predict_proba(x)[:, 1]

    result = original.copy()
    result["backorder_risk_probability"] = probabilities
    result["risk_level"] = [probability_to_risk(float(p), medium_threshold, high_threshold) for p in probabilities]
    result["estimated_monthly_demand_proxy"] = [estimate_monthly_demand(row) for _, row in normalised.iterrows()]
    result["estimated_stock_cover_days"] = [estimate_stock_cover_days(row) for _, row in normalised.iterrows()]
    # Indicative reorder quantity. IMPORTANT: this assumes lead_time is measured in DAYS.
    # If the source uses weeks, multiply lead_time by 7 before relying on this quantity.
    reorder_quantities = []
    for _, row in normalised.iterrows():
        demand = estimate_monthly_demand(row)
        stock = _first_numeric(row, ("national_inv", "current_inventory", "inventory", "on_hand", "stock_on_hand"))
        transit = _first_numeric(row, ("in_transit_qty", "in_transit_qtry", "quantity_in_transit")) or 0.0
        lead = _first_numeric(row, ("lead_time", "supplier_lead_time"))
        buffer = _first_numeric(row, ("min_bank", "minimum_stock", "safety_stock")) or 0.0
        if demand is None or stock is None or lead is None or lead < 0:
            reorder_quantities.append(np.nan)
        else:
            daily_demand = demand / 30.0
            target_stock = daily_demand * lead + max(buffer, 0.0)
            reorder_quantities.append(int(max(0, np.ceil(target_stock - stock - max(transit, 0.0)))))
    result["suggested_reorder_quantity_assuming_lead_time_days"] = reorder_quantities
    result["recommended_action"] = [
        make_recommendation(row, float(p), medium_threshold, high_threshold)
        for (_, row), p in zip(normalised.iterrows(), probabilities)
    ]
    result["recommended_priority_rank"] = result["backorder_risk_probability"].rank(method="first", ascending=False).astype(int)
    result["prediction_target"] = f"Probability of {bundle.get('target_column', 'backorder')}=1 in the dataset's label definition"
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def write_json(data: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")
