import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
# Load data
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler',  StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, X.columns)
])
# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('knn',          KNeighborsRegressor())
])
# Hyperparameter tuning
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights':     ['uniform', 'distance'],
    'knn__p':           [1, 2]
}
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='r2', n_jobs=1
)
grid_search.fit(X_train, y_train)
# Evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Best params: {grid_search.best_params_}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred) ** 0.5:.4f}")
# Save
joblib.dump(best_model, 'california_knn_pipeline.pkl')
print("Model saved.")