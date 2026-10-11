"""Train and compare ML models for salary prediction."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load data
df = pd.read_csv("salarydata.csv")
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])

X, y = df.drop(columns="salary"), df["salary"]
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# 2. Preprocessing (impute + scale numbers, impute + one-hot encode categories)
preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Compare models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {
        "CV R2": cv_r2,
        "Test R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
    }
res_df = pd.DataFrame(results).T
print("\nModel comparison:\n", res_df.round(3))

# 4. Tune the best tree model
best_name = res_df["Test R2"].idxmax()
print("\nBest baseline:", best_name)
pipe = Pipeline([("prep", preprocess),
                 ("model", GradientBoostingRegressor(random_state=42))])
grid = GridSearchCV(pipe, {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3, 4],
}, cv=3, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
pred = best.predict(X_test)
print("Best params:", grid.best_params_)
print(f"Tuned  R2: {r2_score(y_test, pred):.3f} | "
      f"MAE: Rs {mean_absolute_error(y_test, pred):,.0f} | "
      f"RMSE: Rs {np.sqrt(mean_squared_error(y_test, pred)):,.0f}")

# 5. Save model
joblib.dump(best, "salary_model.joblib")
print("Saved model -> salary_model.joblib")

# 6. Plots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y_test, pred, alpha=0.4)
ax[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax[0].set(xlabel="Actual salary", ylabel="Predicted salary", title="Actual vs Predicted")

names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(best.named_steps["model"].feature_importances_, index=names)
imp.nlargest(10).sort_values().plot.barh(ax=ax[1], title="Top 10 feature importances")
plt.tight_layout(); plt.savefig("results.png", dpi=120)
print("Saved plot -> results.png")