import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score

selected_features_class = ['time_spend_company', 'satisfaction_level', 'number_project', 'average_montly_hours', 'salary']
selected_features_reg = ['time_spend_company', 'number_project', 'average_montly_hours', 'salary']
numerical_cols_class = ['time_spend_company', 'satisfaction_level', 'number_project', 'average_montly_hours']
numerical_cols_reg = ['time_spend_company', 'number_project', 'average_montly_hours']
ordinal_cols = ['salary']

salary_categories = ['low', 'medium', 'high']
ordinal_encoder = OrdinalEncoder(categories=[salary_categories])

preprocessor_class = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols_class),
        ('ord', ordinal_encoder, ordinal_cols)
    ]
)

preprocessor_reg = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols_reg),
        ('ord', ordinal_encoder, ordinal_cols)
    ]
)

classification_pipeline = Pipeline([
    ('preprocessor', preprocessor_class),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

regression_pipeline = Pipeline([
    ('preprocessor', preprocessor_reg),
    ('regressor', XGBRegressor(random_state=42))
])

def train_classification_model(df: pd.DataFrame, label: str) -> Pipeline:
    X = df[selected_features_class]
    y = df[label]
    X = X.fillna(X.median(numeric_only=True)).fillna("Unknown")
    classification_pipeline.fit(X, y)
    joblib.dump(classification_pipeline, 'classification_model.pkl')
    return classification_pipeline

def train_regression_model(df: pd.DataFrame) -> Pipeline:
    X = df[selected_features_reg]
    y = df['satisfaction_level']
    X = X.fillna(X.median(numeric_only=True)).fillna("Unknown")
    regression_pipeline.fit(X, y)
    joblib.dump(regression_pipeline, 'regression_model.pkl')
    return regression_pipeline

def predict_probabilities(request: dict, pipeline: Pipeline) -> dict:
    model = pipeline
    input_data = pd.DataFrame([request])
    probability = model.predict_proba(input_data)[0, 1]
    prediction = model.predict(input_data)[0]
    return {"prediction": bool(prediction), "probability": float(probability)}

def predict_satisfaction_level(request: dict, pipeline: Pipeline) -> float:
    model = pipeline
    input_data = pd.DataFrame([request])
    satisfaction_level = model.predict(input_data)[0]
    return float(satisfaction_level)

def plot_left_rate_by_projects(df: pd.DataFrame):
    grouped = df.groupby('number_project')['left'].mean()
    count = df.groupby('number_project')['left'].count()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=grouped.index, y=grouped.values, color='#6495ed')
    plt.xlabel('Number of Projects')
    plt.ylabel('Attrition Rate')
    plt.title('Attrition Rate by Number of Projects')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('static/leave_rate_by_projects.png')
    plt.close()

def evaluate_classification_model(pipeline: Pipeline, df: pd.DataFrame, label: str) -> tuple:
    X = df[selected_features_class]
    y = df[label]
    X = X.fillna(X.median(numeric_only=True)).fillna("Unknown")
    y_pred = pipeline.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return accuracy, f1
