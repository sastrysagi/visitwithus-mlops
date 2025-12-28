import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="VisitWithUs - Wellness Package Purchase Predictor", layout="wide")

MODEL_PATH = Path("artifacts/model.joblib")  # make sure this exists in Space

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def safe_predict(df: pd.DataFrame):
    """
    Predict with helpful error message if feature mismatch happens.
    """
    try:
        pred = model.predict(df)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0][1]
        return pred, proba, None
    except Exception as e:
        return None, None, e

# ---------- UI ----------
st.title("VisitWithUs - Wellness Package Purchase Predictor")
st.write("Enter customer and interaction details to predict whether the customer will purchase the package (**ProdTaken**).")

# Dropdown options (as requested)
TYPE_OF_CONTACT = ["Self Enquiry", "Company Invited"]
OCCUPATION = ["Salaried", "Free Lancer", "Small Business", "Large Business"]
GENDER = ["Male", "Female"]
PRODUCT_PITCHED = ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
PREFERRED_PROPERTY_STAR = [1, 2, 3, 4, 5]
MARITAL_STATUS = ["Married", "Single"]
YES_NO = ["Yes", "No"]
DESIGNATION = ["AVP", "VP", "Manager", "Senior Manager", "Executive"]

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1, format="%d")
    TypeofContact = st.selectbox("TypeofContact", TYPE_OF_CONTACT)
    DurationOfPitch = st.number_input("DurationOfPitch", min_value=0, max_value=10000, value=90, step=1, format="%d")
    Gender_val = st.selectbox("Gender", GENDER)
    NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=1000, value=0, step=1, format="%d")
    PreferredPropertyStar = st.selectbox("PreferredPropertyStar", PREFERRED_PROPERTY_STAR)
    NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, max_value=1000, value=25, step=1, format="%d")
    PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=0, max_value=5, value=5, step=1, format="%d")
    NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=20, value=0, step=1, format="%d")
    MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, max_value=10**10, value=138985000, step=1000, format="%d")

with col2:
    CityTier = st.number_input("CityTier", min_value=1, max_value=3, value=3, step=1, format="%d")
    Occupation_val = st.selectbox("Occupation", OCCUPATION)
    NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=20, value=1, step=1, format="%d")
    ProductPitched_val = st.selectbox("ProductPitched", PRODUCT_PITCHED)
    MaritalStatus_val = st.selectbox("MaritalStatus", MARITAL_STATUS)
    Passport_ui = st.selectbox("Passport", YES_NO)
    OwnCar_ui = st.selectbox("OwnCar", YES_NO)
    Designation_val = st.selectbox("Designation", DESIGNATION)

# Map Yes/No -> 1/0
Passport = 1 if Passport_ui == "Yes" else 0
OwnCar = 1 if OwnCar_ui == "Yes" else 0

# IMPORTANT:
# Your model expects CustomerID. We won't show it, but we will add a dummy value.
CustomerID = 0

# Build input row with EXACT column names expected by training
X = pd.DataFrame([{
    "CustomerID": int(CustomerID),
    "Age": int(Age),
    "TypeofContact": TypeofContact,
    "CityTier": int(CityTier),
    "DurationOfPitch": int(DurationOfPitch),
    "Occupation": Occupation_val,
    "Gender": Gender_val,
    "NumberOfPersonVisiting": int(NumberOfPersonVisiting),
    "NumberOfFollowups": int(NumberOfFollowups),
    "ProductPitched": ProductPitched_val,
    "PreferredPropertyStar": int(PreferredPropertyStar),
    "MaritalStatus": MaritalStatus_val,
    "NumberOfTrips": int(NumberOfTrips),
    "Passport": int(Passport),
    "PitchSatisfactionScore": int(PitchSatisfactionScore),
    "OwnCar": int(OwnCar),
    "NumberOfChildrenVisiting": int(NumberOfChildrenVisiting),
    "Designation": Designation_val,
    "MonthlyIncome": int(MonthlyIncome),
}])

# Optional: enforce a stable column order (helps prevent weird column ordering issues)
FEATURE_ORDER = [
    "CustomerID",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]
X = X[FEATURE_ORDER]

st.divider()

if st.button("Predict"):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model load failed. Ensure `{MODEL_PATH}` exists in your Space.\n\n{e}")
        st.stop()

    pred, proba, err = safe_predict(X)

    if err is not None:
        st.error(
            "Prediction failed. Please ensure UI feature names and data types match the training pipeline exactly.\n\n"
            f"{type(err).__name__}: {err}"
        )
        st.subheader("Debug: Input sent to model")
        st.dataframe(X)
        st.stop()

    label = "Will Purchase (ProdTaken=1)" if int(pred) == 1 else "Will NOT Purchase (ProdTaken=0)"
    st.success(f"Prediction: **{label}**")
    if proba is not None:
        st.info(f"Purchase Probability: **{proba:.3f}**")
