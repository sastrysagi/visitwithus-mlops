import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="VisitWithUs - Package Purchase Predictor", layout="centered")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "model.joblib"

st.title("VisitWithUs - Wellness Package Purchase Predictor")
st.write("Enter customer and interaction details to predict whether the customer will purchase the package (**ProdTaken**).")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Build simple UI from expected columns
# NOTE: These should match the training data columns (excluding target)
feature_columns = [
    "CustomerID","Age","TypeofContact","CityTier","DurationOfPitch","Occupation","Gender",
    "NumberOfPersonVisiting","NumberOfFollowups","ProductPitched","PreferredPropertyStar",
    "MaritalStatus","NumberOfTrips","Passport","PitchSatisfactionScore","OwnCar",
    "NumberOfChildrenVisiting","Designation","MonthlyIncome"
]

with st.form("predict_form"):
    inputs = {}
    col1, col2 = st.columns(2)
    for i, col in enumerate(feature_columns):
        container = col1 if i % 2 == 0 else col2
        with container:
            # Heuristics: numeric vs categorical
            if col in {"TypeofContact","Occupation","Gender","ProductPitched","MaritalStatus","Designation"}:
                inputs[col] = st.text_input(col, value="")
            else:
                inputs[col] = st.number_input(col, value=0.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    df = pd.DataFrame([inputs])
    proba = model.predict_proba(df)[:, 1][0]
    pred = int(proba >= 0.5)

    st.subheader("Prediction")
    st.write(f"**Probability of purchase (ProdTaken=1):** {proba:.3f}")
    st.write(f"**Predicted class:** {'Will Purchase (1)' if pred==1 else 'Will Not Purchase (0)'}")
