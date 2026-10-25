"""Improved Streamlit dashboard for explainable inventory risk decisions."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    load_model_bundle,
    normalise_dataframe,
    score_records,
)

st.set_page_config(
    page_title="Inventory Risk Decision Support",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_bundle(model_path: str, model_mtime: float):
    """Load once per model version; reload automatically after retraining."""
    del model_mtime
    return load_model_bundle(model_path)


@st.cache_data(show_spinner=False, max_entries=3)
def read_csv_cached(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Cache CSV parsing by content so Streamlit reruns don't reread the upload."""
    del file_name
    try:
        return pd.read_csv(BytesIO(file_bytes), low_memory=False)
    except Exception:
        return pd.read_csv(BytesIO(file_bytes), sep=None, engine="python", low_memory=False)


@st.cache_data(show_spinner=False, max_entries=3)
def read_local_csv(path: str, modified_time: float) -> pd.DataFrame:
    """Cache local CSV parsing; file modification time invalidates the cache."""
    del modified_time
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False, max_entries=3)
def cached_score_records(
    records: pd.DataFrame,
    model_path: str,
    model_mtime: float,
    medium_threshold: float,
    high_threshold: float,
    batch_size: int,
) -> pd.DataFrame:
    """Score in batches and cache results across UI reruns."""
    del model_mtime
    bundle = load_model_bundle(model_path)
    if records.empty:
        return score_records(records, bundle, medium_threshold, high_threshold)

    pieces: list[pd.DataFrame] = []
    total = len(records)
    for start in range(0, total, batch_size):
        chunk = records.iloc[start : start + batch_size]
        part = score_records(chunk, bundle, medium_threshold, high_threshold)
        # score_records should preserve order; restore original labels for reliable lookup.
        part.index = records.index[start : start + len(chunk)]
        pieces.append(part)
    return pd.concat(pieces, axis=0)


def format_pct(value: Any) -> str:
    try:
        number = float(value)
        return f"{number:.1%}" if np.isfinite(number) else "—"
    except (TypeError, ValueError):
        return "—"


def safe_float(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
        return f"{number:.{digits}f}" if np.isfinite(number) else "unavailable"
    except (TypeError, ValueError):
        return "unavailable"


def try_shap_explanation(bundle: dict, record: pd.DataFrame) -> pd.DataFrame | None:
    """Optional SHAP explanation for a single record; normal scoring doesn't depend on SHAP."""
    try:
        import shap  # optional dependency
        from src.common import cast_feature_types

        pipeline = bundle["pipeline"]
        feature_columns = bundle["feature_columns"]
        numeric_columns = bundle["numeric_columns"]
        categorical_columns = bundle["categorical_columns"]

        normalised = normalise_dataframe(record)
        raw = normalised.drop(columns=[bundle.get("target_column", "")], errors="ignore")
        x = cast_feature_types(
            raw.reindex(columns=feature_columns),
            numeric_columns,
            categorical_columns,
        )
        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        transformed = preprocessor.transform(x)
        names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)

        if isinstance(values, list):
            if len(values) < 2:
                return None
            contribution = np.asarray(values[1])[0]
        else:
            values_array = np.asarray(values)
            if values_array.ndim == 3:
                contribution = values_array[0, :, 1]
            elif values_array.ndim == 2:
                contribution = values_array[0]
            else:
                return None

        aggregated: dict[str, float] = {}
        for feature, value in zip(names, contribution):
            source_feature = str(feature).split("__", 1)[-1]
            for raw_name in sorted(feature_columns, key=len, reverse=True):
                if source_feature == raw_name or source_feature.startswith(raw_name + "_"):
                    source_feature = raw_name
                    break
            aggregated[source_feature] = aggregated.get(source_feature, 0.0) + float(value)

        result = pd.DataFrame(
            [
                {"feature": name, "impact_toward_backorder": impact}
                for name, impact in aggregated.items()
            ]
        )
        if result.empty:
            return None
        order = result["impact_toward_backorder"].abs().sort_values(ascending=False).index
        return result.loc[order].head(10)
    except Exception:
        return None


st.title("📦 Explainable Inventory Risk Decision Support")
st.caption(
    "Training BOP backorder model • risk-based prioritisation • "
    "transparent inventory recommendations"
)

with st.sidebar:
    st.header("Settings")
    model_path_input = st.text_input("Model file", value=str(DEFAULT_MODEL_PATH))
    model_path = Path(model_path_input).expanduser()
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    st.subheader("Risk categories")
    medium_threshold = st.slider(
        "Medium risk from", min_value=0.05, max_value=0.75,
        value=0.25, step=0.05,
    )
    high_threshold = st.slider(
        "High risk from", min_value=0.30, max_value=0.95,
        value=0.60, step=0.05,
    )
    batch_size = st.select_slider(
        "Scoring batch size",
        options=[500, 1000, 2500, 5000, 10000],
        value=2500,
        help="Smaller batches use less peak memory; larger batches may be faster.",
    )

if high_threshold <= medium_threshold:
    st.error("The high-risk threshold must be greater than the medium-risk threshold.")
    st.stop()

if not model_path.exists():
    st.error("No trained model was found at the selected path.")
    st.markdown(
        "Check that training completed and that the model file exists. "
        "The expected default is usually `models/best_model.joblib`."
    )
    st.code(
        "python src/train.py --data data/Training_BOP.csv\n"
        "python -m streamlit run app.py",
        language="powershell",
    )
    st.caption(f"Checked path: `{model_path}`")
    st.stop()

try:
    model_mtime = model_path.stat().st_mtime
    bundle = get_bundle(str(model_path.resolve()), model_mtime)
except Exception as exc:
    st.error(f"The saved model could not be loaded: {type(exc).__name__}: {exc}")
    st.info("Check the model path or retrain the model.")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("Model information")
    st.write(f"**Selected model:** {bundle.get('selected_model', 'Unknown')}")
    st.write(f"**Training source:** {bundle.get('training_source_name', 'Unknown')}")
    st.write(f"**Target:** `{bundle.get('target_column', 'Unknown')}`")
    rows_used = bundle.get("training_rows_used", "Unknown")
    try:
        rows_used = f"{int(rows_used):,}"
    except (TypeError, ValueError):
        pass
    st.write(f"**Rows used for training:** {rows_used}")
    st.write(
        f"**Model decision threshold:** "
        f"{float(bundle.get('decision_threshold', 0.5)):.3f}"
    )

st.warning(
    "This model estimates the historical Training BOP label `went_on_backorder`. "
    "Do not describe it as a verified 14–30-day stockout forecast unless the target's "
    "time horizon is confirmed. Recommendations are decision support, not purchase orders."
)

st.subheader("1. Choose inventory data")
example_path = ROOT / "data" / "Training_BOP.csv"
sources = ["Upload a CSV"]
if example_path.exists():
    sources.insert(0, "Use data/Training_BOP.csv")

source = st.radio("Data source", sources, horizontal=True)
records: pd.DataFrame | None = None

if source == "Use data/Training_BOP.csv":
    try:
        records = read_local_csv(str(example_path), example_path.stat().st_mtime)
    except Exception as exc:
        st.error(f"Could not read {example_path.name}: {exc}")
else:
    uploaded = st.file_uploader(
        "Upload a cleaned inventory CSV",
        type=["csv"],
        help="Large CSVs can use substantial memory. Try a smaller sample if your computer struggles.",
    )
    if uploaded is not None:
        try:
            # Parse directly from Streamlit's uploaded-file buffer instead of
            # creating an extra full-file bytes copy with getvalue().
            with st.spinner("Reading your CSV..."):
                uploaded.seek(0)
                records = pd.read_csv(uploaded, low_memory=False)
        except Exception as exc:
            st.error(
                f"Could not read the uploaded CSV: "
                f"{type(exc).__name__}: {exc}"
            )

if records is None:
    st.info("Choose the project dataset or upload a CSV to score inventory records.")
    st.stop()

if records.empty or len(records.columns) == 0:
    st.warning("The selected CSV contains no usable rows or columns.")
    st.stop()

st.caption(f"Loaded **{len(records):,} rows** and **{len(records.columns):,} columns**.")
with st.expander("Preview input data"):
    st.dataframe(records.head(20), use_container_width=True, hide_index=True)

st.subheader("2. Score inventory records")
try:
    with st.spinner(
        f"Scoring {len(records):,} records in batches of {batch_size:,}… "
        "The first run may take longer."
    ):
        scored = cached_score_records(
            records,
            str(model_path.resolve()),
            model_mtime,
            medium_threshold,
            high_threshold,
            int(batch_size),
        )
except Exception as exc:
    st.error(f"Could not score the selected records: {type(exc).__name__}: {exc}")
    st.write(
        "Check that the CSV has compatible inventory features. Missing fields may be imputed, "
        "but unrelated columns, different units, or different meanings can make predictions unreliable."
    )
    st.stop()

required_score_columns = {"backorder_risk_probability", "risk_level"}
if not required_score_columns.issubset(scored.columns):
    st.error("The scoring function did not return expected risk columns. Check src/common.py.")
    st.stop()

total = len(scored)
high_count = int((scored["risk_level"] == "High").sum())
medium_count = int((scored["risk_level"] == "Medium").sum())
low_count = int((scored["risk_level"] == "Low").sum())
avg_probability = float(scored["backorder_risk_probability"].mean()) if total else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records scored", f"{total:,}")
c2.metric("High risk", f"{high_count:,}")
c3.metric("Medium risk", f"{medium_count:,}")
c4.metric("Mean estimated risk", format_pct(avg_probability))

st.subheader("3. Risk overview")
distribution = scored["risk_level"].value_counts().reindex(
    ["High", "Medium", "Low"], fill_value=0
)
chart_col, guidance_col = st.columns([1, 2])
with chart_col:
    st.bar_chart(distribution.rename("Records"))
with guidance_col:
    st.markdown("**Interpreting the scores**")
    st.write(
        "The probability is a model estimate for the historical backorder label, "
        "not a guarantee that a product will run out of stock."
    )
    st.write(
        "The monthly demand proxy and reorder quantity are heuristics, not a trained "
        "demand forecast or validated replenishment policy."
    )
    st.warning(
        "Confirm the units of `lead_time`, `national_inv`, `in_transit_qty`, and `min_bank` "
        "before using reorder quantities. The current heuristic assumes lead time is in days "
        "and uses `min_bank` as a simple buffer."
    )

st.subheader("4. Products requiring attention")
filter_col, search_col = st.columns([1, 2])
with filter_col:
    risk_filter = st.multiselect(
        "Risk levels to show", ["High", "Medium", "Low"], default=["High", "Medium"]
    )
with search_col:
    search_text = st.text_input(
        "Optional product / ID search",
        placeholder="Search a SKU, product ID, or item ID",
    )

filtered = scored[scored["risk_level"].isin(risk_filter)].copy()
if search_text.strip():
    search_columns = [
        col for col in filtered.columns
        if str(col).strip().lower() in {"sku", "product_id", "item_id"}
    ]
    if search_columns:
        mask = pd.Series(False, index=filtered.index)
        for col in search_columns:
            mask |= filtered[col].astype(str).str.contains(
                search_text.strip(), case=False, na=False, regex=False
            )
        filtered = filtered[mask]

sort_columns = [
    col for col in (
        "backorder_risk_probability",
        "suggested_reorder_quantity_assuming_lead_time_days",
    )
    if col in filtered.columns
]
if sort_columns:
    filtered = filtered.sort_values(
        sort_columns, ascending=[False] * len(sort_columns), na_position="last"
    )

visible_cols: list[str] = []
for candidate in (
    "sku", "product_id", "item_id", "national_inv", "lead_time",
    "in_transit_qty", "in_transit_qtry", "went_on_backorder",
):
    matches = [col for col in filtered.columns if str(col).strip().lower() == candidate]
    if matches:
        visible_cols.append(matches[0])
visible_cols += [
    "backorder_risk_probability", "risk_level",
    "estimated_monthly_demand_proxy", "estimated_stock_cover_days",
    "suggested_reorder_quantity_assuming_lead_time_days", "recommended_action",
]
visible_cols = list(dict.fromkeys(col for col in visible_cols if col in filtered.columns))

if filtered.empty:
    st.info("No records match the selected risk levels and search.")
else:
    display_limit = st.selectbox(
        "Rows displayed on screen",
        options=[100, 500, 1000, 5000],
        index=1,
        help="Only the visible table is limited; the download includes all scored rows.",
    )
    st.caption(f"Showing {min(len(filtered), display_limit):,} of {len(filtered):,} matching records.")
    st.dataframe(
        filtered[visible_cols].head(display_limit),
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "Download all scored records (CSV)",
    data=scored.to_csv(index=False).encode("utf-8-sig"),
    file_name="scored_inventory_risk.csv",
    mime="text/csv",
)

st.subheader("5. Explain an individual score")
if filtered.empty:
    st.info("Select at least one risk level with matching records.")
else:
    # Use source row positions for stable mapping from the scored output to input data.
    row_position_by_index: dict[Any, int] = {}
    for pos, label in enumerate(records.index):
        row_position_by_index.setdefault(label, pos)
    selectable = filtered.copy()
    selectable["_source_position"] = [row_position_by_index.get(idx, -1) for idx in selectable.index]
    selectable = selectable[selectable["_source_position"] >= 0]

    if selectable.empty:
        st.info("Could not map these scored records back to their original input rows.")
    else:
        selected_index = st.selectbox(
            "Choose a record",
            options=selectable.index.tolist(),
            format_func=lambda idx: (
                f"Input row {int(selectable.loc[idx, '_source_position']) + 1:,} · "
                f"{selectable.loc[idx, 'risk_level']} · "
                f"{float(selectable.loc[idx, 'backorder_risk_probability']):.1%} risk"
            ),
        )
        selected = selectable.loc[selected_index]
        source_position = int(selected["_source_position"])
        left, right = st.columns([1, 2])
        with left:
            st.metric("Estimated backorder risk", format_pct(selected["backorder_risk_probability"]))
            st.write(f"**Risk level:** {selected['risk_level']}")
            if "estimated_monthly_demand_proxy" in selected.index:
                st.write(
                    f"**Monthly demand proxy:** "
                    f"{safe_float(selected['estimated_monthly_demand_proxy'])} units/month"
                )
            if "estimated_stock_cover_days" in selected.index:
                st.write(
                    f"**Estimated stock cover:** "
                    f"{safe_float(selected['estimated_stock_cover_days'])} days"
                )
        with right:
            st.markdown("**Recommended management action**")
            st.write(selected.get("recommended_action", "Review inventory and supplier information."))
            st.caption(
                "These rules are transparent decision-support heuristics. Verify stock counts, "
                "supplier conditions, demand, and commercial constraints before acting."
            )

        with st.expander("Inspect original input values"):
            input_row = records.iloc[[source_position]]
            st.dataframe(
                input_row.T.rename(columns={input_row.index[0]: "Value"}),
                use_container_width=True,
            )

        with st.expander("Optional local SHAP explanation (requires `pip install shap`)"):
            if st.button("Calculate explanation for this record"):
                with st.spinner("Calculating the local explanation…"):
                    shap_result = try_shap_explanation(bundle, records.iloc[[source_position]])
                if shap_result is None or shap_result.empty:
                    st.info(
                        "SHAP is unavailable or incompatible with this model. "
                        "Use global feature importance as a faster alternative."
                    )
                else:
                    st.caption(
                        "Positive contributions push toward the backorder class; negative "
                        "contributions push away. These are model explanations, not causes."
                    )
                    st.dataframe(shap_result, use_container_width=True, hide_index=True)

with st.expander("6. Model evaluation and global feature importance"):
    metrics = bundle.get("test_metrics", {})
    if metrics:
        metric_columns = st.columns(5)
        for column, label, key in zip(
            metric_columns,
            ["Precision", "Recall", "F1", "F2", "ROC-AUC"],
            ["precision", "recall", "f1", "f2", "roc_auc"],
        ):
            column.metric(label, format_pct(metrics.get(key)))
        st.caption(
            "Metrics come from a held-out evaluation split. For imbalanced backorder data, "
            "review recall, precision, false negatives, and alert volume together."
        )
        with st.expander("Detailed saved metrics"):
            st.json({key: value for key, value in metrics.items() if key != "confusion_matrix"})
    else:
        st.info("No saved evaluation metrics were found in this model bundle.")

    importances = pd.DataFrame(bundle.get("feature_importance", []))
    if not importances.empty and {"feature", "importance"}.issubset(importances.columns):
        st.markdown("**Global predictor importance**")
        st.caption(
            "Importance shows which fields the model relies on overall; it is not causal "
            "and does not explain one specific record."
        )
        st.bar_chart(importances.head(15).set_index("feature")["importance"])
    else:
        st.info("No global feature-importance report is saved in this model bundle.")

st.divider()
st.caption(
    "Academic prototype. Not connected to a live inventory system and does not place orders "
    "automatically. Validate model performance and inventory assumptions before operational use."
)
