import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telecom Customer Churn Predictor", layout="centered")
st.title("Telecom Customer Churn Predictor")
st.write("Enter customer details to estimate the probability of churn.")


@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")


@st.cache_data
def load_feature_template():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=["customerID", "Churn"])
    # Take one valid row as a clean template for all remaining columns
    return df.iloc[[0]].copy()


model = load_model()
input_df = load_feature_template()

tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input(
    "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0
)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("Predict Churn"):
    row = input_df.copy()
    row["tenure"] = tenure
    row["MonthlyCharges"] = monthly_charges
    row["Contract"] = contract
    row["InternetService"] = internet_service
    if "TotalCharges" in row.columns and pd.isna(row["TotalCharges"].iloc[0]):
        row["TotalCharges"] = tenure * monthly_charges

    churn_prob = float(model.predict_proba(row)[0, 1])
    st.subheader(f"Churn Probability: {churn_prob:.1%}")
    if churn_prob >= 0.5:
        st.warning("High risk of churn — consider retention offers.")
    else:
        st.success("Low risk of churn.")