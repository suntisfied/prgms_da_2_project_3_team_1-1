from fastapi import FastAPI, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from model.pipeline import train_classification_model, train_regression_model, predict_probabilities, predict_satisfaction_level, plot_left_rate_by_projects, evaluate_classification_model

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

selected_features_class = ['time_spend_company', 'satisfaction_level', 'number_project', 'average_montly_hours', 'salary']
selected_features_reg = ['time_spend_company', 'number_project', 'average_montly_hours', 'salary']
numerical_cols_class = ['time_spend_company', 'satisfaction_level', 'number_project', 'average_montly_hours']
numerical_cols_reg = ['time_spend_company', 'number_project', 'average_montly_hours']
ordinal_cols = ['salary']

classification_pipeline = None
regression_pipeline = None

class PredictionRequest(BaseModel):
    satisfaction_level: float
    time_spend_company: float
    number_project: float
    average_montly_hours: float
    salary: str

@app.post("/train/")
async def train(file: UploadFile, label: str = Form(...)):
    global classification_pipeline, regression_pipeline
    try:
        df = pd.read_csv(file.file)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if label not in df.columns:
        raise HTTPException(status_code=400, detail=f"Label column '{label}' not found in dataset.")

    classification_pipeline = train_classification_model(df, label)
    regression_pipeline = train_regression_model(df)

    feature_importances_class = classification_pipeline.named_steps['classifier'].feature_importances_
    all_feature_names_class = numerical_cols_class + ordinal_cols
    importance_df_class = pd.DataFrame({'Feature': all_feature_names_class, 'Importance': feature_importances_class})
    importance_df_class = importance_df_class.sort_values(by='Importance', ascending=False)
    importance_df_class['Feature'] = importance_df_class['Feature'].replace('time_spend_company', 'Job Tenure (years)')

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df_class, color='#6495ed')
    plt.title('Feature Importance for XGBoost Classification Model')
    plt.tight_layout()
    plt.savefig('static/feature_importance_class.png')
    plt.close()

    feature_importances_reg = regression_pipeline.named_steps['regressor'].feature_importances_
    all_feature_names_reg = numerical_cols_reg + ordinal_cols
    importance_df_reg = pd.DataFrame({'Feature': all_feature_names_reg, 'Importance': feature_importances_reg})
    importance_df_reg = importance_df_reg.sort_values(by='Importance', ascending=False)
    importance_df_reg['Feature'] = importance_df_reg['Feature'].replace('time_spend_company', 'Job Tenure (years)')

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df_reg, color='#6495ed')
    plt.title('Feature Importance for XGBoost Regression Model')
    plt.tight_layout()
    plt.savefig('static/feature_importance_reg.png')
    plt.close()

    plot_left_rate_by_projects(df)

    accuracy, f1 = evaluate_classification_model(classification_pipeline, df, label)

    return {
        "message": "Models trained successfully using XGBoost",
        "classification_graph_url": "/static/feature_importance_class.png",
        "regression_graph_url": "/static/feature_importance_reg.png",
        "leave_rate_by_projects_url": "/static/leave_rate_by_projects.png",
        "accuracy": accuracy,
        "f1_score": f1
    }

@app.post("/predict/")
async def predict(request: PredictionRequest):
    prediction_result = predict_probabilities(request.dict(), classification_pipeline)
    satisfaction_level = predict_satisfaction_level(request.dict(), regression_pipeline)

    probability = float(prediction_result['probability']) * 100
    if prediction_result['prediction']:
        recommendations = f"This employee has a {probability:.2f}% probability of attrition. To reduce this probability, it's crucial to ensure employee satisfaction with the company. Increasing the number of projects from 3 to 5 can significantly enhance satisfaction, thereby lowering the likelihood of attrition."
    else:
        recommendations = "No attrition is expected for this employee."

    return {
        "prediction": prediction_result['prediction'],
        "probability": probability,
        "recommendations": recommendations
    }

@app.get("/", response_class=HTMLResponse)
async def serve_home_page(request: Request):
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
