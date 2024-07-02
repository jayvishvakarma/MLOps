import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

logged_model = 'model.pkl'
model = joblib.load(logged_model)

categorical_features = ['employment_type', 'job_category', 'experience_level',
                        'employee_residence', 'remote_ratio', 'company_location', 'company_size']

distinct_values = {
    'experience_level': ['Senior-level/Expert','Mid-level/Intermediate', 'Entry-level/Junior'], 
    'employment_type': ['Full-time', 'Contractor', 'Freelancer', 'Part-time'],  
    'employee_residence': ['ES', 'US', 'CA', 'DE', 'GB', 'NG', 'IN', 'HK', 'PT', 'NL', 'CH', 'CF', 'FR', 'AU',
 'FI', 'UA', 'IE', 'IL', 'GH', 'AT', 'CO', 'SG', 'SE', 'SI', 'MX', 'UZ', 'BR', 'TH',
 'HR', 'PL', 'KW', 'VN', 'CY', 'AR', 'AM', 'BA', 'KE', 'GR', 'MK', 'LV', 'RO', 'PK',
 'IT', 'MA', 'LT', 'BE', 'AS', 'IR', 'HU', 'SK', 'CN', 'CZ', 'CR', 'TR', 'CL', 'PR',
 'DK', 'BO', 'PH', 'DO', 'EG', 'ID', 'AE', 'MY', 'JP', 'EE', 'HN', 'TN', 'RU', 'DZ',
 'IQ', 'BG', 'JE', 'RS', 'NZ', 'MD', 'LU', 'MT'], 
    'remote_ratio': ['Full-Remote', 'On-Site', 'Half-Remote'], 
    'company_location': ['ES', 'US', 'CA', 'DE', 'GB', 'NG', 'IN', 'HK', 'NL', 'CH', 'CF', 'FR', 'FI', 'UA',
 'IE', 'IL', 'GH', 'CO', 'SG', 'AU', 'SE', 'SI', 'MX', 'BR', 'PT', 'RU', 'TH', 'HR',
 'VN', 'EE', 'AM', 'BA', 'KE', 'GR', 'MK', 'LV', 'RO', 'PK', 'IT', 'MA', 'PL', 'AL',
 'AR', 'LT', 'AS', 'CR', 'IR', 'BS', 'HU', 'AT', 'SK', 'CZ', 'TR', 'PR', 'DK', 'BO',
 'PH', 'BE', 'ID', 'EG', 'AE', 'LU', 'MY', 'HN', 'JP', 'DZ', 'IQ', 'CN', 'NZ', 'CL',
 'MD', 'MT'],  
    'company_size': ['LARGE', 'SMALL', 'MEDIUM'], 
    'job_category': ['Other', 'Machine Learning', 'Data Science', 'Data Engineering',
 'Data Architecture', 'Management'] 
}

encoders = {feature: LabelEncoder().fit(values) for feature, values in distinct_values.items()}

st.title("Salary Prediction")

user_input = {}
for feature in categorical_features:
    user_input[feature] = st.selectbox(f"Select {feature}",distinct_values[feature])

encoded_input = [encoders[feature].transform([user_input[feature]])[0] for feature in categorical_features]

if st.button("Predict Salary Range"):
    encoded_input = np.array(encoded_input).reshape(1, -1)
    prediction = model.predict(encoded_input)

    salary_labels = ['low', 'low-mid', 'mid', 'mid-high', 'high', 'very-high', 'Top']
   
    st.write(f"Predicted Salary Range: {prediction}")
 