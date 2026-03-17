import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import sys
import traceback

print(f"Python version: {sys.version}", file=sys.stderr)
print(f"Current directory: {Path.cwd()}", file=sys.stderr)
print(f"Files in directory: {list(Path.cwd().glob('*.pkl'))}", file=sys.stderr)

st.set_page_config(page_title="California Housing Predictor", layout="centered")
st.title('California Housing Price Predictor')
st.write('Enter the details below to get a predicted house value.')


def load_model():
    """Load the trained KNN model."""
    preferred_file = Path('california_knn_pipeline.pkl')
    print(f"Looking for model file: {preferred_file.absolute()}", file=sys.stderr)
    
    if preferred_file.exists():
        print(f"Found model file: {preferred_file}", file=sys.stderr)
        return joblib.load(preferred_file)

    fallback_files = sorted(Path('.').glob('california_knn_pipeline*.pkl'))
    if fallback_files:
        print(f"Found fallback model file: {fallback_files[0]}", file=sys.stderr)
        return joblib.load(fallback_files[0])

    available_files = list(Path('.').glob('*'))
    raise FileNotFoundError(
        f"No model file found. Available files: {available_files}\n"
        "Run train.py or ensure 'california_knn_pipeline.pkl' exists."
    )


try:
    print("Attempting to load model...", file=sys.stderr)
    model = load_model()
    print("Model loaded successfully!", file=sys.stderr)
except Exception as e:
    error_msg = f"Failed to load model: {str(e)}\n\n{traceback.format_exc()}"
    print(error_msg, file=sys.stderr)
    st.error(error_msg)
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

    
