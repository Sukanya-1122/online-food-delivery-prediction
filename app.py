"""Streamlit interface for online food delivery order prediction."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).parent / "models" / "food_delivery_model.pkl"

st.set_page_config(page_title="Online Food Delivery Order Prediction", page_icon="🍽️", layout="centered")
st.title("Online Food Delivery Order Prediction")
st.caption("Enter customer details to predict whether the customer is likely to order online food delivery.")


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run train_model.py before starting Streamlit.")
    return joblib.load(MODEL_PATH)


try:
    artifact = load_artifact()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

options = artifact["categorical_options"]
defaults = artifact["numeric_defaults"]
ranges = artifact["numeric_ranges"]

with st.form("prediction_form"):
    st.subheader("Customer details")
    left, right = st.columns(2)

    with left:
        age = st.number_input("Age", min_value=0, max_value=120, value=int(defaults["Age"]), step=1)
        gender = st.selectbox("Gender", options["Gender"])
        marital_status = st.selectbox("Marital Status", options["Marital Status"])
        occupation = st.selectbox("Occupation", options["Occupation"])
        monthly_income = st.selectbox("Monthly Income", options["Monthly Income"])
        educational_qualifications = st.selectbox(
            "Educational Qualifications", options["Educational Qualifications"]
        )

    with right:
        family_size = st.number_input(
            "Family size", min_value=1, max_value=50, value=int(defaults["Family size"]), step=1
        )
        customer_type = st.selectbox("Customer Type", options["Customer Type"])
        latitude = st.number_input(
            "latitude", min_value=-90.0, max_value=90.0, value=float(defaults["latitude"]), format="%.6f"
        )
        longitude = st.number_input(
            "longitude", min_value=-180.0, max_value=180.0, value=float(defaults["longitude"]), format="%.6f"
        )
        pin_code = st.number_input(
            "Pin code", min_value=0, max_value=999999, value=int(defaults["Pin code"]), step=1
        )
        feedback = st.selectbox("Feedback", options["Feedback"])

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    try:
        input_data = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": gender,
                    "Marital Status": marital_status,
                    "Occupation": occupation,
                    "Monthly Income": monthly_income,
                    "Educational Qualifications": educational_qualifications,
                    "Family size": family_size,
                    "Customer Type": customer_type,
                    "latitude": latitude,
                    "longitude": longitude,
                    "Pin code": pin_code,
                    "Feedback": feedback,
                }
            ],
            columns=artifact["feature_columns"],
        )
        prediction = str(artifact["model"].predict(input_data)[0])
        if prediction not in {"Yes", "No"}:
            raise ValueError("The model returned an unexpected output.")

        st.success(f"Predicted Output: {prediction}")
        if prediction == "Yes":
            st.write("Yes -> Customer is predicted to use/order online food delivery.")
        else:
            st.write("No -> Customer is predicted not to use/order online food delivery.")
    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Prediction could not be completed: {error}")
