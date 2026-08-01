import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

st.set_page_config(
    page_title="Bank Marketing Model Evaluator",
    page_icon="🏦",
    layout="wide",
)

model_options = {
    "Logistic Regression": "logistic",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
}

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
    selected_model_name = st.selectbox("Select model", list(model_options.keys()))

st.title("🏦 Bank Marketing Model Evaluator")

with st.container():
    st.markdown(
        f"<h3 style='color:#1f77b4; font-weight:700; margin-bottom:0.2rem;'>Selected Model: {selected_model_name}</h3>",
        unsafe_allow_html=True,
    )

st.write(
    "Upload a test CSV file to evaluate a trained bank marketing classifier."
)
st.caption(
    "If your uploaded file contains a deposit column, it will be used as the target. Otherwise, the app will use the built-in encoded test labels."
)


@st.cache_data
def load_default_data():
    X_test = pd.read_csv("data/X_test_encoded.csv")
    y_test = pd.read_csv("data/y_test_encoded.csv").iloc[:, 0]
    return X_test, y_test


@st.cache_resource
def load_model(model_name):
    return joblib.load(f"models/{model_name}.pkl")


def prepare_features(df, expected_columns):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                cleaned = df[col].astype(str).str.strip().str.lower()
                if set(cleaned.dropna().unique()) <= {"true", "false", "1", "0"}:
                    df[col] = cleaned.map({"true": True, "false": False, "1": True, "0": False})
            except Exception:
                pass

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[expected_columns]


if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        default_X, default_y = load_default_data()
        expected_columns = default_X.columns.tolist()

        if "deposit" in uploaded_df.columns:
            y_true = uploaded_df["deposit"].astype(int)
            X_eval = uploaded_df.drop(columns=["deposit"])
            source_label = uploaded_file.name
        else:
            y_true = default_y.iloc[: len(uploaded_df)].reset_index(drop=True)
            X_eval = uploaded_df
            source_label = f"{uploaded_file.name} (using built-in labels)"

        X_eval = prepare_features(X_eval, expected_columns)
    except Exception as exc:
        st.error(f"Unable to process the uploaded file: {exc}")
        st.stop()
else:
    default_X, default_y = load_default_data()
    X_eval = default_X.copy()
    y_true = default_y.copy()
    source_label = "default encoded test data"

model_key = model_options[selected_model_name]
model = load_model(model_key)
predictions = model.predict(X_eval)

accuracy = accuracy_score(y_true, predictions)
precision = precision_score(y_true, predictions, zero_division=0)
recall = recall_score(y_true, predictions, zero_division=0)
f1 = f1_score(y_true, predictions, zero_division=0)
cm = confusion_matrix(y_true, predictions)
classification_rep = classification_report(
    y_true,
    predictions,
    target_names=["No", "Yes"],
    output_dict=True,
)

st.success(
    f"Evaluation complete: {selected_model_name} was applied to {len(X_eval)} rows from {source_label}."
)

st.info(
    f"Model: {selected_model_name} | Data source: {source_label} | Rows evaluated: {len(X_eval)}"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1 Score", f"{f1:.3f}")

st.subheader("Confusion Matrix")
st.dataframe(
    pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Predicted No", "Predicted Yes"])
)

st.subheader("Classification Report")
st.dataframe(pd.DataFrame(classification_rep).T)
