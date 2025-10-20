import logging
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Simple, robust Loan Approval Prediction Streamlit app
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

st.title("🏦 Loan Approval Predictor")
st.write("Simple app to explore loan dataset, train a RandomForest, and make predictions.")

# ---------------------------
# Utility functions
# ---------------------------

@st.cache_data
def load_data_from_file(path: str):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.exception("Failed to load CSV")
        return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    return df

def default_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    expected = [
        "no_of_dependents", "education", "self_employed",
        "income_annum", "loan_amount", "loan_term", "cibil_score",
        "bank_asset_value", "luxury_assets_value",
        "residential_assets_value", "commercial_assets_value", "loan_status"
    ]
    present = [c for c in expected if c in df.columns]
    df = df[present].copy()

    if "bank_asset_value" in df.columns or "luxury_assets_value" in df.columns:
        bank = df["bank_asset_value"] if "bank_asset_value" in df.columns else pd.Series(0, index=df.index)
        lux = df["luxury_assets_value"] if "luxury_assets_value" in df.columns else pd.Series(0, index=df.index)
        df["movable_assets"] = bank.fillna(0) + lux.fillna(0)
    else:
        df["movable_assets"] = 0

    if "residential_assets_value" in df.columns or "commercial_assets_value" in df.columns:
        res = df["residential_assets_value"] if "residential_assets_value" in df.columns else pd.Series(0, index=df.index)
        com = df["commercial_assets_value"] if "commercial_assets_value" in df.columns else pd.Series(0, index=df.index)
        df["immovable_assets"] = res.fillna(0) + com.fillna(0)
    else:
        df["immovable_assets"] = 0

    if "education" in df.columns:
        df["education"] = df["education"].astype(str).str.lower().map({"graduate": 1, "not_graduate": 0}).fillna(
            df["education"].apply(lambda x: 1 if str(x).strip() == "1" else 0)
        ).astype(int)

    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(
            df["self_employed"].apply(lambda x: 1 if str(x).strip() == "1" else 0)
        ).astype(int)

    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].astype(str).str.lower().map({"approved": 1, "rejected": 0}).fillna(
            df["loan_status"].apply(lambda x: 1 if str(x).strip() == "1" else 0)
        ).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df

def get_model_features(df: pd.DataFrame):
    features = [
        "no_of_dependents", "education", "self_employed",
        "income_annum", "loan_amount", "loan_term",
        "cibil_score", "movable_assets", "immovable_assets"
    ]
    return [f for f in features if f in df.columns]

# ---------------------------
# Data load / upload
# ---------------------------

st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload loan CSV (optional)", type=["csv"])
if uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    sample_path = "loan_approval_dataset.csv"
    df_raw = load_data_from_file(sample_path)
    if df_raw is None:
        st.sidebar.info("No default CSV found. Upload your dataset.")
        st.stop()

df = default_preprocess(df_raw)
st.sidebar.success(f"Loaded dataset with {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------
# Navigation
# ---------------------------

page = st.sidebar.selectbox("Page", ["Home", "Data", "Train", "Predict"])

if page == "Home":
    st.header("Overview")
    st.write("This app trains a RandomForest classifier to predict loan approval.")
    st.write("Features:", get_model_features(df))
    st.write("Target: loan_status (1=Approved, 0=Rejected)")
    st.dataframe(df.head())

elif page == "Data":
    st.header("Data Overview")
    st.write(df.describe())
    if "loan_status" in df.columns:
        st.bar_chart(df["loan_status"].value_counts())

elif page == "Train":
    st.header("Train RandomForest Model")

    if "loan_status" not in df.columns:
        st.error("No 'loan_status' target column found.")
        st.stop()

    features = get_model_features(df)
    if not features:
        st.error("No compatible features found.")
        st.stop()

    test_size = st.slider("Test set proportion", 0.1, 0.5, 0.2)
    n_estimators = st.slider("Trees", 50, 300, 100, step=50)
    random_state = st.number_input("Random state", value=42, step=1)

    if st.button("Train model"):
        X = df[features]
        y = df["loan_status"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

        model = RandomForestClassifier(n_estimators=int(n_estimators), random_state=int(random_state), n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"Model trained successfully! Accuracy: {acc:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        with open("loan_model.pkl", "wb") as f:
            pickle.dump({"model": model, "features": features}, f)
        st.success("Model saved as loan_model.pkl")

elif page == "Predict":
    st.header("🔮 Loan Approval Prediction")

    with st.form("predict_form"):
        no_of_dependents = st.slider("Dependents", 0, 5, 2)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        income_annum = st.number_input("Annual Income", min_value=0, value=500000)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=150000)
        loan_term = st.slider("Loan Term (Years)", 1, 20, 10)
        cibil_score = st.slider("CIBIL Score", 300, 900, 750)
        residential_assets = st.number_input("Residential Assets", min_value=0, value=100000)
        commercial_assets = st.number_input("Commercial Assets", min_value=0, value=100000)
        luxury_assets = st.number_input("Luxury Assets", min_value=0, value=150000)
        bank_assets = st.number_input("Bank Assets", min_value=0, value=150000)
        submit = st.form_submit_button("Predict")

    if submit:
        movable_assets = bank_assets + luxury_assets
        immovable_assets = residential_assets + commercial_assets

        input_data = {
            "no_of_dependents": no_of_dependents,
            "education": 1 if education == "Graduate" else 0,
            "self_employed": 1 if self_employed == "Yes" else 0,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "movable_assets": movable_assets,
            "immovable_assets": immovable_assets,
        }

        input_df = pd.DataFrame([input_data])

        try:
            with open("loan_model.pkl", "rb") as f:
                saved = pickle.load(f)
            model = saved["model"]
            features = saved["features"]

            for feat in features:
                if feat not in input_df.columns:
                    input_df[feat] = 0
            input_df = input_df[features]

            # Make prediction (safe handling of predict_proba)
            prediction = model.predict(input_df)[0]

            # Safely extract probability
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                if len(proba) == 2:
                    prob = proba[1]  # Probability of Approved (class 1)
                else:
                    # Only one class in model
                    prob = proba[0]
            else:
                prob = 0.0

            if prediction == 1:
                st.success(f"✅ LOAN APPROVED (Probability: {prob:.2f})")
            else:
                st.error(f"❌ LOAN REJECTED (Probability: {prob:.2f})")

        except FileNotFoundError:
            st.warning("Train the model first before prediction.")
