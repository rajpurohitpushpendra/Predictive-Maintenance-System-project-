import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

rf_model = joblib.load("models/random_forest.pkl")
xgb_model = joblib.load("models/xgboost.pkl")

st.title("⚙️ Predictive Maintenance System")
st.write("Predict machine failure using industrial machine sensor parameters.")

st.divider()

st.subheader("Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    air_temperature = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=298.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=308.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=500,
        max_value=3000,
        value=1500,
        step=10
    )

with col2:
    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )

    model_choice = st.selectbox(
        "Select Model",
        ["XGBoost", "Random Forest"]
    )

st.divider()

if st.button("Predict Machine Failure", use_container_width=True):

    temperature_difference = process_temperature - air_temperature

    if model_choice == "XGBoost":
        input_data = pd.DataFrame(
            [[
                air_temperature,
                process_temperature,
                temperature_difference,
                rotational_speed,
                torque,
                tool_wear
            ]],
            columns=[
                "air_temperature",
                "process_temperature",
                "temperature_difference",
                "rotational_speed",
                "torque",
                "tool_wear"
            ]
        )

        prediction = xgb_model.predict(input_data)[0]
        probability = xgb_model.predict_proba(input_data)[0][1]

    else:
        input_data = pd.DataFrame(
            [[
                air_temperature,
                process_temperature,
                temperature_difference,
                rotational_speed,
                torque,
                tool_wear
            ]],
            columns=[
                "Air temperature [K]",
                "Process temperature [K]",
                "Temperature difference",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]
        )

        prediction = rf_model.predict(input_data)[0]
        probability = rf_model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Machine Failure Predicted")
    else:
        st.success("✅ No Machine Failure Predicted")

    st.metric(
        "Failure Probability",
        f"{probability * 100:.2f}%"
    )

st.divider()

st.subheader("Model Performance")

performance = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": ["98.25%", "98.75%"],
    "Precision": ["74.63%", "87.72%"],
    "Recall": ["73.53%", "73.53%"],
    "F1 Score": ["74.07%", "80.00%"]
})

st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True
)