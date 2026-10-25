"""Streamlit dashboard for the Training BOP inventory decision-support prototype."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.common import DEFAULT_MODEL_PATH, load_model_bundle, normalise_dataframe, score_records  # noqa: E402

st.set_page_config(page_title="Inventory Risk Decision Support", page_icon="📦", layout="wide")


@st.cache_resource(show_spinner=False)
def get_bundle(model_path: str):
    return load_model_bundle(model_path)


def format_pct(value) -> str:
    try:
        return f"{float(value):.1%}" if value is not None else "—"
    except (TypeError, ValueError):
        return "—"


def try_shap_explanation(bundle, record: pd.DataFrame) -> pd.DataFrame | None:
    """Optional local SHAP explanation. Returns None if SHAP is unavailable or incompatible."""
    try:
        import shap  # optional dependency
        from src.common import cast_feature_types
        pipeline = bundle["pipeline"]
        feature_columns = bundle["feature_columns"]
        numeric_columns = bundle["numeric_columns"]
        categorical_columns = bundle["categorical_columns"]
        normalised = normalise_dataframe(record)
        raw = normalised.drop(columns=[bundle.get("target_column", "")], errors="ignore")
        x = cast_feature_types(raw.reindex(columns=feature_columns), numeric_columns, categorical_columns)
        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        transformed = preprocessor.transform(x)
        names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
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
        result = pd.DataFrame([{"feature": k, "impact_toward_backorder": v} for k, v in aggregated.items()])
        return result.reindex(result["impact_toward_backorder"].abs().sort_values(ascending=False).index).head(10)
    except Exception:
        return None


st.title("📦 Explainable Inventory Risk Decision Support")
st.caption("Training BOP dataset • saved Random Forest/classifier bundle • explainable backorder-risk recommendations")

model_path = st.sidebar.text_input("Model file", value=str(DEFAULT_MODEL_PATH))
try:
    bundle = get_bundle(model_path)
except Exception as exc:
    st.error("No trained model is available yet.")
    st.markdown("Put your **cleaned Training BOP CSV** in `data/Training_BOP.csv`, then train the model from the project terminal:")
    st.code("python src/eda.py --data data/Training_BOP.csv\npython src/train.py --data data/Training_BOP.csv\nstreamlit run app.py", language="powershell")
    st.caption(f"Details: {exc}")
    st.stop()

with st.sidebar:
    st.subheader("Risk categories")
    medium_threshold = st.slider("Medium risk from", min_value=0.05, max_value=0.75, value=0.25, step=0.05)
    high_threshold = st.slider("High risk from", min_value=0.30, max_value=0.95, value=0.60, step=0.05)
    if high_threshold <= medium_threshold:
        st.warning("High-risk threshold should be greater than the medium-risk threshold.")
        st.stop()
    st.divider()
    st.subheader("Model information")
    st.write(f"**Selected model:** {bundle.get('selected_model', 'Unknown')}")
    st.write(f"**Training source:** {bundle.get('training_source_name', 'Unknown')}")
    st.write(f"**Target:** `{bundle.get('target_column', 'Unknown')}`")
    st.write(f"**Rows used for training:** {bundle.get('training_rows_used', 'Unknown'):,}")
    st.write(f"**Validation alert threshold:** {bundle.get('decision_threshold', 0.5):.3f}")

st.warning(
    "Target limitation: this model estimates the Training BOP label `went_on_backorder`. "
    "It should not be described as a verified 14–30-day stockout forecast unless the cleaned data's "
    "time-linked target definition confirms that horizon. Recommendations are decision support, not automatic purchase orders."
)

st.subheader("Score inventory records")
example_path = ROOT / "data" / "Training_BOP.csv"
if example_path.exists():
    st.caption("A dataset file was detected at `data/Training_BOP.csv`; upload a different cleaned/current snapshot below to score it, or use the detected file.")
    use_detected = st.checkbox("Use data/Training_BOP.csv", value=False)
else:
    use_detected = False
uploaded = st.file_uploader("Upload a cleaned inventory CSV to score", type=["csv"])
records: pd.DataFrame | None = None
if uploaded is not None:
    try:
        records = pd.read_csv(uploaded, low_memory=False)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
elif use_detected and example_path.exists():
    try:
        records = pd.read_csv(example_path, low_memory=False)
    except Exception as exc:
        st.error(f"Could not read {example_path.name}: {exc}")

if records is None:
    st.info("Upload the cleaned BOP/inventory CSV to view risk scores and recommendations. The original training file is not packaged with this project.")
else:
    try:
        with st.spinner("Scoring inventory records…"):
            scored = score_records(records, bundle, medium_threshold, high_threshold)
        total = len(scored)
        high_count = int((scored["risk_level"] == "High").sum())
        medium_count = int((scored["risk_level"] == "Medium").sum())
        avg_probability = float(scored["backorder_risk_probability"].mean()) if total else 0.0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Products / records", f"{total:,}")
        c2.metric("High risk", f"{high_count:,}")
        c3.metric("Medium risk", f"{medium_count:,}")
        c4.metric("Mean predicted probability", format_pct(avg_probability))

        st.subheader("Risk overview")
        distribution = scored["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        col_chart, col_help = st.columns([1, 2])
        with col_chart:
            st.bar_chart(distribution.rename("Records"))
        with col_help:
            st.markdown("**How to read the scores**")
            st.write("The probability is a model estimate for the dataset's historical backorder label, not a guarantee that a product will run out of stock.")
            st.write("Monthly demand proxy uses the highest available per-month rate among recent sales and existing 3/6/9-month forecasts; it is a heuristic, not a trained forecast.")
            st.warning("Suggested reorder quantity assumes lead_time is in DAYS and uses min_bank as a simple buffer. If lead_time is in WEEKS, multiply it by 7 before relying on the quantity. Confirm units and safety-stock policy first.")
            st.write("Risk bands are adjustable in the sidebar and should be reviewed with inventory managers against the consequences of missed risks and unnecessary alerts.")
            st.write("Negative stock-cover estimates can occur when recorded inventory is negative. Treat these as a stock-deficit warning; they are not literal negative days.")

        st.subheader("Products requiring attention")
        risk_filter = st.multiselect("Show risk levels", ["High", "Medium", "Low"], default=["High", "Medium"])
        filtered = scored[scored["risk_level"].isin(risk_filter)].copy()
        filtered = filtered.sort_values(["backorder_risk_probability", "suggested_reorder_quantity_assuming_lead_time_days"], ascending=[False, False], na_position="last")
        visible_cols = []
        for candidate in ("sku", "product_id", "item_id", "national_inv", "lead_time", "in_transit_qty", "in_transit_qtry", "went_on_backorder"):
            # Use the source's original casing when present.
            matches = [c for c in filtered.columns if str(c).strip().lower() == candidate]
            if matches:
                visible_cols.append(matches[0])
        visible_cols += ["backorder_risk_probability", "risk_level", "estimated_monthly_demand_proxy", "estimated_stock_cover_days", "suggested_reorder_quantity_assuming_lead_time_days", "recommended_action"]
        visible_cols = list(dict.fromkeys(c for c in visible_cols if c in filtered.columns))
        st.dataframe(filtered[visible_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download all scored records (CSV)",
            data=scored.to_csv(index=False).encode("utf-8-sig"),
            file_name="scored_inventory_risk.csv",
            mime="text/csv",
        )

        st.subheader("Explain an individual score")
        choices = filtered.index.tolist()
        if not choices:
            st.write("No records match the selected risk filter.")
        else:
            selected_position = st.selectbox(
                "Choose a record", options=choices,
                format_func=lambda idx: f"Row {idx} · {scored.loc[idx, 'risk_level']} · {scored.loc[idx, 'backorder_risk_probability']:.1%} risk"
            )
            selected = scored.loc[selected_position]
            left, right = st.columns([1, 2])
            with left:
                st.metric("Backorder probability", format_pct(selected["backorder_risk_probability"]))
                st.write(f"**Risk level:** {selected['risk_level']}")
                st.write(f"**Monthly demand proxy:** {selected['estimated_monthly_demand_proxy']:.1f} units/month" if pd.notna(selected["estimated_monthly_demand_proxy"]) else "**Monthly demand proxy:** unavailable from available columns")
                st.write(f"**Estimated stock cover:** {selected['estimated_stock_cover_days']:.1f} days" if pd.notna(selected["estimated_stock_cover_days"]) else "**Estimated stock cover:** unavailable from available columns")
            with right:
                st.markdown("**Recommended management action**")
                st.write(selected["recommended_action"])
                st.caption("Recommendation rules are transparent inventory heuristics. Confirm stock counts, supplier conditions, demand, and commercial constraints before acting.")

            with st.expander("Inspect input values"):
                input_row = records.iloc[[selected_position]] if isinstance(selected_position, (int, np.integer)) and selected_position < len(records) else records.loc[[selected_position]]
                st.dataframe(input_row.T.rename(columns={input_row.index[0]: "Value"}), use_container_width=True)

            with st.expander("Optional local SHAP explanation (requires `pip install shap`)"):
                if st.button("Calculate SHAP explanation for this record"):
                    # Use source row position where the scored dataframe retains the same row order.
                    try:
                        local_record = records.iloc[[int(selected_position)]]
                    except Exception:
                        local_record = records.loc[[selected_position]]
                    shap_result = try_shap_explanation(bundle, local_record)
                    if shap_result is None or shap_result.empty:
                        st.info("SHAP is not installed or does not support this model/preprocessing combination. Use the global feature-importance report and the inventory checks above instead.")
                    else:
                        st.caption("Positive values push the model output toward the backorder class; negative values push away from it. These are model contributions, not causal explanations.")
                        st.dataframe(shap_result, use_container_width=True, hide_index=True)

        with st.expander("Model evaluation"):
            metrics = bundle.get("test_metrics", {})
            if metrics:
                metric_columns = st.columns(5)
                for column, label, key in zip(metric_columns,
                    ["Precision", "Recall", "F1", "F2", "ROC-AUC"],
                    ["precision", "recall", "f1", "f2", "roc_auc"]):
                    column.metric(label, format_pct(metrics.get(key)))
                st.caption("Metrics are from a held-out evaluation split. Because backorders are commonly rare, review recall, precision, false negatives, and alert volume together.")
                st.json({k: v for k, v in metrics.items() if k != "confusion_matrix"})
            else:
                st.info("No saved evaluation metrics found in the model bundle.")

            importances = pd.DataFrame(bundle.get("feature_importance", []))
            if not importances.empty and {"feature", "importance"}.issubset(importances.columns):
                st.markdown("**Global predictor importance**")
                st.caption("Global importance indicates which fields contributed most to model performance; it does not explain a specific prediction or establish causation.")
                st.bar_chart(importances.head(15).set_index("feature")["importance"])
    except Exception as exc:
        st.error(f"Could not score the uploaded records: {type(exc).__name__}: {exc}")
        st.write("Check that the CSV has compatible inventory features. Missing training features are imputed, but incorrect units or different meanings will reduce reliability.")

st.divider()
st.caption("Prototype for academic evaluation. Not connected to a live inventory system and does not place orders automatically.")
