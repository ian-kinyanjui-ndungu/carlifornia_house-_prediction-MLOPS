import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.title('California Housing Price Predictor')
st.write('Enter the details below to get a predicted house value.')


def load_model():
    preferred_file = Path('california_knn_pipeline.pkl')
    if preferred_file.exists():
        return joblib.load(preferred_file)

    fallback_files = sorted(Path('.').glob('california_knn_pipeline*.pkl'))
    if fallback_files:
        return joblib.load(fallback_files[0])

    raise FileNotFoundError(
        "No model file found. Run train.py or place a file named "
        "'california_knn_pipeline.pkl' in this folder."
    )


try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

#Creating  user input fields


st.subheader('Property details')

col1, col2 = st.columns(2)
with col1:
    MedInc     = st.number_input('Median Income',      value=3.5)
    HouseAge   = st.number_input('House Age',           value=25.0)
    AveRooms   = st.number_input('Average Rooms',       value=5.2)
    AveBedrms  = st.number_input('Average Bedrooms',    value=1.1)
with col2:
    Population = st.number_input('Population',          value=1200.0)
    AveOccup   = st.number_input('Average Occupancy',   value=2.8)
    Latitude   = st.number_input('Latitude',            value=34.1)
    Longitude  = st.number_input('Longitude',           value=-118.3)

# Prediction logic
if st.button('Predict House Value'):
    input_data = pd.DataFrame([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]], columns=model.feature_names_in_)
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: ${prediction[0] * 100:.0f},000")

    
