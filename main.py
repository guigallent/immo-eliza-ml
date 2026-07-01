import pandas as pd
import numpy as np
import joblib
from src.config import RANDOM_STATE
from src.clean import clean_data
from src.preprocess import preprocess_data
from src.train import split_data, scale_data, linear_regression, decision_tree_regression, random_forest_regression, xgboost_regression
from src.evaluate import evaluate_model

df = pd.read_csv("./data/properties_database.csv")

df_cleaned = clean_data(df)

X_train, X_test, y_train, y_test = split_data(df_cleaned, test_size=0.2, random_state=RANDOM_STATE)

X_train_processed, fitted_encoders = preprocess_data(X_train)
X_test_processed, _ = preprocess_data(X_test, encoders=fitted_encoders)

X_train_scaled, X_test_scaled, scaler = scale_data(X_train_processed, X_test_processed)

linear = linear_regression(X_train_scaled, y_train)
joblib.dump(linear, "models/linear.joblib")
print("Linear Regression model saved as 'models/linear.joblib'.")

decision_tree = decision_tree_regression(X_train_scaled, y_train, max_depth=8)
joblib.dump(decision_tree, "models/decision_tree.joblib")
print("Decision Tree Regressor model saved as 'models/decision_tree.joblib'.")

random_forest = random_forest_regression(X_train_scaled, y_train, n_estimators=200)
joblib.dump(random_forest, "models/random_forest.joblib")
print("Random Forest Regressor model saved as 'models/random_forest.joblib'.")

xgboost = xgboost_regression(X_train_scaled, y_train, n_estimators=200, max_depth=6, learning_rate=0.05)
joblib.dump(xgboost, "models/xgboost.joblib")
print("XGBoost Regressor model saved as 'models/xgboost.joblib'.")

#evaluate_model(model, X_test_scaled, y_test)