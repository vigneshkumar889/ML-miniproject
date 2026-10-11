"""Load the trained model and predict a salary for a new person."""
import joblib
import pandas as pd

model = joblib.load("salary_model.joblib")

new_person = pd.DataFrame([{
    "age": 28, "gender": "Female", "education": "Masters",
    "job_title": "Data Scientist", "experience_years": 4.0,
    "location": "Chennai", "company_size": "Large", "skills_count": 7,
}])
print(f"Predicted annual salary: Rs {model.predict(new_person)[0]:,.0f}")