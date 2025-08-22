"""
Auto ML Report Web App (Streamlit)
=================================

Features
--------
- Upload CSV, XLSX, JSON, Parquet, or MATLAB .mat (2-D) files.
- Choose target column and task type (Auto / Regression / Classification).
- Select one or multiple models and set hyperparameters from the UI.
- End‑to‑end sklearn Pipelines with preprocessing (impute, encode, scale).
- Metrics & plots for regression/classification.
- One‑click PDF report with metrics, parameters, and figures.

How to run
----------
1) Create a virtual env (recommended) and install requirements:

   pip install -r requirements.txt

2) Start the app:

   streamlit run app.py

3) Open the URL shown in terminal.

Create requirements.txt (sample)
--------------------------------
Put this in a file named requirements.txt next to app.py:

streamlit>=1.33
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
matplotlib>=3.8
fpdf2>=2.7
scipy>=1.11
pyarrow>=15.0
xgboost>=2.0

Notes
-----
- For large .mat files with complex structures, only 2-D numeric arrays are supported; you can export tables from MATLAB as CSV for best results.
- XGBoost is optional; deselect it if not available.
"""

import io
import os
import json
import math
import tempfile
import datetime as dt
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from fpdf import FPDF

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC

# Optional models
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import pyarrow as _  # noqa
    PARQUET_OK = True
except Exception:
    PARQUET_OK = False

st.set_page_config(page_title="Auto ML Report App", layout="wide")
st.title("📊 Auto ML Report — Upload • Train • Evaluate • Export PDF")

# -------------------------
# Utilities
# -------------------------

def load_data(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(uploaded_file)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    if suffix in [".json"]:
        return pd.read_json(uploaded_file)
    if suffix in [".parquet"]:
        if not PARQUET_OK:
            raise RuntimeError("pyarrow not installed; can't read Parquet. Remove Parquet or install pyarrow.")
        return pd.read_parquet(uploaded_file)
    if suffix in [".mat"]:
        from scipy.io import loadmat
        mat = loadmat(uploaded_file)
        # Heuristic: pick the first 2-D numeric array not starting with __
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
                # Create column labels c0..cn
                cols = [f"c{i}" for i in range(v.shape[1])]
                return pd.DataFrame(v, columns=cols)
        raise ValueError("No 2-D numeric array found in .mat file. Please export a table or CSV from MATLAB.")
    raise ValueError(f"Unsupported file type: {suffix}")


def infer_task_type(y: pd.Series) -> str:
    # Auto: numeric -> regression; else classification
    if pd.api.types.is_numeric_dtype(y):
        # If small integer cardinality, may still be classification; give rule of thumb
        nunique = y.nunique(dropna=True)
        if nunique <= max(10, int(0.02 * len(y))):
            return "classification"
        return "regression"
    return "classification"


def split_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str], scale_numeric: bool) -> ColumnTransformer:
    num_steps = []
    if len(num_cols) > 0:
        num_steps = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
    cat_steps = []
    if len(cat_cols) > 0:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps) if num_steps else "drop", num_cols),
            ("cat", Pipeline(cat_steps) if cat_steps else "drop", cat_cols),
        ]
    )
    return preprocessor


def regression_models_ui() -> Dict[str, Dict]:
    models = {}
    with st.expander("Linear Regression", expanded=True):
        use = st.checkbox("Use LinearRegression", value=True, key="linreg_use")
        if use:
            models["LinearRegression"] = {}
    with st.expander("Random Forest Regressor", expanded=False):
        use = st.checkbox("Use RandomForestRegressor", value=True, key="rfr_use")
        if use:
            n_estimators = st.number_input("n_estimators", 50, 2000, 300, step=50, key="rfr_n")
            max_depth = st.number_input("max_depth (0=none)", 0, 200, 0, step=1, key="rfr_d")
            models["RandomForestRegressor"] = {"n_estimators": int(n_estimators), "max_depth": None if max_depth == 0 else int(max_depth), "n_jobs": -1, "random_state": 42}
    with st.expander("XGBoost Regressor", expanded=False):
        if not XGB_AVAILABLE:
            st.info("XGBoost not installed; add it to requirements to enable.")
        use = st.checkbox("Use XGBRegressor", value=False, key="xgbr_use", disabled=not XGB_AVAILABLE)
        if use:
            lr = st.number_input("learning_rate", 0.001, 1.0, 0.1, step=0.01, key="xgbr_lr")
            md = st.number_input("max_depth", 1, 20, 6, step=1, key="xgbr_md")
            ne = st.number_input("n_estimators", 50, 3000, 300, step=50, key="xgbr_ne")
            subsample = st.number_input("subsample", 0.1, 1.0, 1.0, step=0.1, key="xgbr_ss")
            colsample = st.number_input("colsample_bytree", 0.1, 1.0, 1.0, step=0.1, key="xgbr_cs")
            models["XGBRegressor"] = {
                "learning_rate": float(lr),
                "max_depth": int(md),
                "n_estimators": int(ne),
                "subsample": float(subsample),
                "colsample_bytree": float(colsample),
                "random_state": 42,
                "n_jobs": -1,
            }
    return models


def classification_models_ui() -> Dict[str, Dict]:
    models = {}
    with st.expander("Logistic Regression", expanded=True):
        use = st.checkbox("Use LogisticRegression", value=True, key="logreg_use")
        if use:
            C = st.number_input("C (inverse regularization)", 0.001, 1000.0, 1.0, step=0.1, key="logreg_C")
            max_iter = st.number_input("max_iter", 100, 5000, 1000, step=100, key="logreg_iter")
            models["LogisticRegression"] = {"C": float(C), "max_iter": int(max_iter), "n_jobs": -1}
    with st.expander("Random Forest Classifier", expanded=False):
        use = st.checkbox("Use RandomForestClassifier", value=True, key="rfc_use")
        if use:
            n_estimators = st.number_input("n_estimators", 50, 2000, 300, step=50, key="rfc_n")
            max_depth = st.number_input("max_depth (0=none)", 0, 200, 0, step=1, key="rfc_d")
            models["RandomForestClassifier"] = {"n_estimators": int(n_estimators), "max_depth": None if max_depth == 0 else int(max_depth), "n_jobs": -1, "random_state": 42}
    with st.expander("Support Vector Classifier", expanded=False):
        use = st.checkbox("Use SVC (with probability)", value=False, key="svc_use")
        if use:
            C = st.number_input("C", 0.001, 1000.0, 1.0, step=0.1, key="svc_C")
            gamma = st.selectbox("gamma", ["scale", "auto"], index=0, key="svc_gamma")
            models["SVC"] = {"C": float(C), "gamma": gamma, "probability": True}
    with st.expander("XGBoost Classifier", expanded=False):
        if not XGB_AVAILABLE:
            st.info("XGBoost not installed; add it to requirements to enable.")
        use = st.checkbox("Use XGBClassifier", value=False, key="xgbc_use", disabled=not XGB_AVAILABLE)
        if use:
            lr = st.number_input("learning_rate", 0.001, 1.0, 0.1, step=0.01, key="xgbc_lr")
            md = st.number_input("max_depth", 1, 20, 6, step=1, key="xgbc_md")
            ne = st.number_input("n_estimators", 50, 3000, 300, step=50, key="xgbc_ne")
            subsample = st.number_input("subsample", 0.1, 1.0, 1.0, step=0.1, key="xgbc_ss")
            colsample = st.number_input("colsample_bytree", 0.1, 1.0, 1.0, step=0.1, key="xgbc_cs")
            models["XGBClassifier"] = {
                "learning_rate": float(lr),
                "max_depth": int(md),
                "n_estimators": int(ne),
                "subsample": float(subsample),
                "colsample_bytree": float(colsample),
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss",
            }
    return models


# -------------------------
# Sidebar — Data & Config
# -------------------------
with st.sidebar:
    st.header("1) Upload your data")
    uploaded = st.file_uploader(
        "Upload CSV / XLSX / JSON / Parquet / MAT",
        type=["csv", "txt", "xlsx", "xls", "json", "parquet", "mat"],
    )

    scale_numeric = st.checkbox("Standardize numeric features", value=False)
    test_size = st.slider("Test size (fraction)", 0.05, 0.5, 0.2, step=0.05)
    random_state = st.number_input("Random state", 0, 999999, 42, step=1)

    st.header("2) Target & Task")
    target_col = None
    task_choice = "Auto"
    if uploaded:
        try:
            df = load_data(uploaded)
            st.session_state["raw_df"] = df
            target_col = st.selectbox("Select target column", df.columns)
            task_choice = st.selectbox("Task type", ["Auto", "Regression", "Classification"], index=0)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

    st.header("3) Choose models")
    chosen_models: Dict[str, Dict] = {}
    if uploaded and target_col:
        X, y, num_cols, cat_cols = split_features(st.session_state["raw_df"], target_col)
        auto_task = infer_task_type(y) if task_choice == "Auto" else task_choice.lower()
        st.caption(f"Inferred task: **{auto_task}**")

        if auto_task == "regression":
            chosen_models = regression_models_ui()
        else:
            chosen_models = classification_models_ui()

    st.header("4) Train!")
    run_btn = st.button("Run training and generate results", type="primary", disabled=not (uploaded and target_col and chosen_models))


# -------------------------
# Main — Results
# -------------------------
if uploaded and target_col:
    df = st.session_state["raw_df"].copy()
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")

    X, y, num_cols, cat_cols = split_features(df, target_col)

    if task_choice == "Auto":
        task = infer_task_type(y)
    else:
        task = task_choice.lower()

    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric)

    if run_btn:
        st.success("Training started. Scroll for results.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if task == "classification" and y.nunique() > 1 else None
        )

        results = []  # collect per-model results
        figures = []  # (title, filepath)

        def save_current_figure(title: str) -> str:
            tmpdir = st.session_state.setdefault("fig_dir", tempfile.mkdtemp(prefix="aml_figs_"))
            fname = os.path.join(tmpdir, f"{title.replace(' ', '_').replace('/', '-')}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
            return fname

        # Train/evaluate per model
        for model_name, params in chosen_models.items():
            if task == "regression":
                if model_name == "LinearRegression":
                    model = LinearRegression(**params)
                elif model_name == "RandomForestRegressor":
                    model = RandomForestRegressor(**params)
                elif model_name == "XGBRegressor" and XGB_AVAILABLE:
                    model = XGBRegressor(**params)
                else:
                    st.warning(f"Model {model_name} not available; skipping.")
                    continue
            else:  # classification
                if model_name == "LogisticRegression":
                    # multi_class handled automatically by lbfgs when y has >2 classes
                    model = LogisticRegression(**params)
                elif model_name == "RandomForestClassifier":
                    model = RandomForestClassifier(**params)
                elif model_name == "SVC":
                    model = SVC(**params)
                elif model_name == "XGBClassifier" and XGB_AVAILABLE:
                    model = XGBClassifier(**params)
                else:
                    st.warning(f"Model {model_name} not available; skipping.")
                    continue

            pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)

            entry = {
                "model": model_name,
                "params": params,
                "task": task,
            }

            if task == "regression":
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = math.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                entry.update({"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})

                # Plots: Pred vs Actual, Residuals
                fig1 = plt.figure()
                plt.scatter(y_test, y_pred, alpha=0.6)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title(f"{model_name} — Predicted vs Actual")
                p1 = save_current_figure(f"{model_name}_Pred_vs_Actual")
                figures.append((f"{model_name} — Predicted vs Actual", p1))

                resid = y_test - y_pred
                fig2 = plt.figure()
                plt.scatter(y_pred, resid, alpha=0.6)
                plt.axhline(0, linestyle="--")
                plt.xlabel("Predicted")
                plt.ylabel("Residual")
                plt.title(f"{model_name} — Residuals vs Predicted")
                p2 = save_current_figure(f"{model_name}_Residuals")
                figures.append((f"{model_name} — Residuals", p2))

            else:  # classification
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted")
                entry.update({"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig3 = plt.figure()
                plt.imshow(cm, aspect="auto")
                plt.title(f"{model_name} — Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                for (i, j), v in np.ndenumerate(cm):
                    plt.text(j, i, int(v), ha='center', va='center')
                p3 = save_current_figure(f"{model_name}_Confusion_Matrix")
                figures.append((f"{model_name} — Confusion Matrix", p3))

                # ROC (binary only) + PR curve
                proba_ok = hasattr(pipe.named_steps["model"], "predict_proba")
                if proba_ok and y_test.nunique() == 2:
                    proba = pipe.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, proba, pos_label=list(pd.Series(y_test).unique())[1])
                    auc = roc_auc_score(y_test, proba)
                    entry["ROC_AUC"] = auc
                    fig4 = plt.figure()
                    plt.plot(fpr, tpr)
                    plt.plot([0, 1], [0, 1], linestyle='--')
                    plt.xlabel("FPR")
                    plt.ylabel("TPR")
                    plt.title(f"{model_name} — ROC Curve (AUC={auc:.3f})")
                    p4 = save_current_figure(f"{model_name}_ROC")
                    figures.append((f"{model_name} — ROC", p4))

                    precs, recs, _ = precision_recall_curve(y_test, proba, pos_label=list(pd.Series(y_test).unique())[1])
                    fig5 = plt.figure()
                    plt.plot(recs, precs)
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title(f"{model_name} — Precision-Recall Curve")
                    p5 = save_current_figure(f"{model_name}_PR")
                    figures.append((f"{model_name} — PR", p5))

            results.append(entry)

        # Show results table
        if results:
            st.subheader("Results summary")
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            st.session_state["results_df"] = res_df
            st.session_state["figures"] = figures
        else:
            st.warning("No results to show.")

        # PDF generation
        if results:
            st.subheader("Generate PDF Report")

            title = st.text_input("Report title", value="Auto ML Report")
            author = st.text_input("Author / Owner", value="")
            include_head = st.checkbox("Include dataset head (first 20 rows) as table", value=True)

            def build_pdf() -> bytes:
                pdf = FPDF(unit="mm", format="A4")
                pdf.set_auto_page_break(auto=True, margin=15)

                def h1(text):
                    pdf.set_font("Arial", "B", 18)
                    pdf.cell(0, 10, txt=text, ln=True)

                def h2(text):
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 8, txt=text, ln=True)

                def p(text):
                    pdf.set_font("Arial", size=11)
                    pdf.multi_cell(0, 6, txt=text)

                # Cover
                pdf.add_page()
                h1(title)
                p(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nAuthor: {author}")

                # Dataset overview
                h2("Dataset Overview")
                p(f"Filename: {uploaded.name}\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
                # Summary stats (numeric only)
                desc = df.describe(include=[np.number]).round(3)
                if not desc.empty:
                    p("Numeric summary (first 8 columns shown if many):")
                    cols = desc.columns[:8]
                    table = desc[cols]
                    # Render as simple text table
                    p(json.dumps(json.loads(table.to_json(orient="split")), indent=2))

                # Target & task
                h2("Target & Task")
                p(f"Target column: {target_col}\nTask type: {task}")

                # Results table
                h2("Model Results")
                res_print = st.session_state["results_df"].copy()
                # Shorten params for print
                res_print["params"] = res_print["params"].apply(lambda d: json.dumps(d)[:200] + ("..." if len(json.dumps(d))>200 else ""))
                # Print as JSON (simple & robust)
                p(json.dumps(json.loads(res_print.to_json(orient="records")), indent=2))

                # Dataset head
                if include_head:
                    h2("Dataset Preview (head)")
                    # Render first 20 rows as text
                    head_txt = df.head(20).to_csv(index=False)
                    p(head_txt)

                # Figures
                if st.session_state.get("figures"):
                    h2("Figures")
                    for title, path in st.session_state["figures"]:
                        pdf.add_page()
                        h2(title)
                        # Maintain width within page
                        max_w = 180
                        pdf.image(path, w=max_w)

                # Return bytes
                return bytes(pdf.output(dest="S").encode("latin1"))

            if st.button("🖨️ Build PDF Report"):
                try:
                    pdf_bytes = build_pdf()
                    st.session_state["pdf_bytes"] = pdf_bytes
                    st.success("PDF generated. Use the button below to download.")
                except Exception as e:
                    st.error(f"Failed to build PDF: {e}")

            if st.session_state.get("pdf_bytes"):
                st.download_button(
                    label="⬇️ Download Report PDF",
                    data=st.session_state["pdf_bytes"],
                    file_name=f"auto_ml_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )

else:
    st.info("Upload a dataset to get started.")
