import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline

# Define the paths to the model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classification_model_path = os.path.join(BASE_DIR, 'classification_model.pkl')
regression_model_path = os.path.join(BASE_DIR, 'regression_model.pkl')

# Load pre-trained models
classification_pipeline = joblib.load(classification_model_path)
regression_pipeline = joblib.load(regression_model_path)

# Define the features used for prediction
selected_features_class = ['time_spend_company', 'satisfaction_level', 'number_project', 'average_montly_hours', 'salary']
selected_features_reg = ['time_spend_company', 'number_project', 'average_montly_hours', 'salary']

def predict_probabilities(input_data: dict) -> dict:
    input_df = pd.DataFrame([input_data])
    probability = classification_pipeline.predict_proba(input_df)[0, 1]
    prediction = classification_pipeline.predict(input_df)[0]
    return {"prediction": bool(prediction), "probability": float(probability)}

def predict_satisfaction_level(input_data: dict) -> float:
    input_df = pd.DataFrame([input_data])
    satisfaction_level = regression_pipeline.predict(input_df)[0]
    return float(satisfaction_level)

def predict_attrition(input_data: dict) -> dict:
    probability_result = predict_probabilities(input_data)
    satisfaction_level = predict_satisfaction_level(input_data)

    probability = probability_result['probability'] * 100
    if probability_result['prediction']:
        recommendations = f"This employee has a {probability:.2f}% probability of attrition. To reduce this probability, it's crucial to ensure employee satisfaction with the company. Increasing the number of projects from 3 to 5 can significantly enhance satisfaction, thereby lowering the likelihood of attrition."
    else:
        recommendations = "No attrition is expected for this employee."

    return {
        "prediction": probability_result['prediction'],
        "probability": probability,
        "recommendations": recommendations,
        "satisfaction_level": satisfaction_level
    }
