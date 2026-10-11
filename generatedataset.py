"""Generates a realistic synthetic salary dataset (annual salary in INR)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000

job_base = {  # base annual salary (INR) for each role
    "Data Analyst": 500000, "Software Engineer": 700000, "Data Scientist": 900000,
    "ML Engineer": 1000000, "DevOps Engineer": 800000, "Product Manager": 1100000,
    "HR Executive": 400000, "Accountant": 380000, "Sales Executive": 350000,
}
edu_mult = {"High School": 0.75, "Bachelors": 1.0, "Masters": 1.2, "PhD": 1.4}
loc_mult = {"Bangalore": 1.20, "Mumbai": 1.15, "Delhi": 1.10, "Hyderabad": 1.10,
            "Chennai": 1.0, "Pune": 1.0, "Kolkata": 0.9}
size_mult = {"Startup": 0.95, "Medium": 1.0, "Large": 1.15}

jobs = rng.choice(list(job_base), N)
edu = rng.choice(list(edu_mult), N, p=[0.05, 0.55, 0.33, 0.07])
loc = rng.choice(list(loc_mult), N)
size = rng.choice(list(size_mult), N, p=[0.3, 0.35, 0.35])
gender = rng.choice(["Male", "Female"], N)
exp = np.clip(rng.gamma(2.5, 2.5, N), 0, 30).round(1)
age = np.clip(22 + exp + rng.normal(2, 2, N), 20, 60).round().astype(int)
skills = np.clip(rng.poisson(4 + exp / 4), 1, 15)

salary = np.array([
    job_base[j] * edu_mult[e] * loc_mult[l] * size_mult[s]
    for j, e, l, s in zip(jobs, edu, loc, size)
])
salary *= (1 + 0.07 * exp) ** 0.9          # experience growth
salary *= 1 + 0.015 * skills               # skills bonus
salary *= rng.normal(1, 0.08, N)           # random noise
salary = salary.round(-3)

df = pd.DataFrame({
    "age": age, "gender": gender, "education": edu, "job_title": jobs,
    "experience_years": exp, "location": loc, "company_size": size,
    "skills_count": skills, "salary": salary.astype(int),
})
# Add a few missing values to practice cleaning
for col in ["education", "experience_years"]:
    df.loc[rng.choice(N, 30, replace=False), col] = np.nan

df.to_csv("salary_data.csv", index=False)
print(df.shape); print(df.head())