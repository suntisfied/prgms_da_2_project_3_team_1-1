from model.predictor import predict_attrition

# Example input data
input_data = {
    "satisfaction_level": 0.38,
    "time_spend_company": 3,
    "number_project": 2,
    "average_montly_hours": 157,
    "salary": "low"
}

# Predict attrition
result = predict_attrition(input_data)

# Print the result
print(result)
