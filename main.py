import pandas as pd
import numpy as np
import joblib
from src.config import RANDOM_STATE
from src.clean import clean_data
from src.preprocess import preprocess_data
from src.train import split_data, scale_data, linear_regression, decision_tree_regression, random_forest_regression, xgboost_regression
from src.evaluate import cross_validate_model, evaluate_model, print_cross_validation_results, print_evaluation

def main():
    """
    Pipeline:
    1. Import the raw data
    2. Clean the data
    3. Split the data into training and testing sets
    4. Preprocess the features (encoding categorical variables, imputing missing values, etc.)
    5. Scale the data sets using StandardScaler
    6. Train multiple regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor, XGBoost Regressor) and save them as .joblib files
    7. Evaluate their performance on the test set (including cross-validation)
    """ 

    # 1. Import the raw data
    df = pd.read_csv("./data/properties_database.csv")

    # 2. Clean the data
    df_cleaned = clean_data(df)

    # 3. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df_cleaned, test_size=0.2, random_state=RANDOM_STATE)

    # 4. Preprocess the features
    X_train_processed, fitted_encoders = preprocess_data(X_train)
    X_test_processed, _ = preprocess_data(X_test, encoders=fitted_encoders)

    fitted_encoders["train_columns"] = list(X_train_processed.columns)
    joblib.dump(fitted_encoders, "artifacts/encoders.joblib")
    print("Fitted encoders saved as 'artifacts/encoders.joblib'.")

    # 5. Scale the data sets using StandardScaler
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train_processed, X_test_processed)
    joblib.dump(scaler, "artifacts/scaler.joblib")
    print("Fitted scaler saved as 'artifacts/scaler.joblib'.")

    # 6. Train multiple regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor, XGBoost Regressor) and save them as .joblib files
    linear = linear_regression(X_train_scaled, y_train)
    joblib.dump(linear, "models/linear.joblib")
    print("Linear Regression model saved as 'models/linear.joblib'.")

    decision_tree = decision_tree_regression(X_train_processed, y_train, min_samples_split=40, min_samples_leaf=16, max_depth=20, ccp_alpha= 0.1)
    joblib.dump(decision_tree, "models/decision_tree.joblib")
    print("Decision Tree Regressor model saved as 'models/decision_tree.joblib'.")

    random_forest = random_forest_regression(X_train_processed, y_train, n_estimators=300, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=30)
    joblib.dump(random_forest, "models/random_forest.joblib")
    print("Random Forest Regressor model saved as 'models/random_forest.joblib'.")

    xgboost = xgboost_regression(X_train_processed, y_train, subsample=0.9, reg_lambda=10, reg_alpha=0, n_estimators=1200, min_child_weight=1, max_depth=7, learning_rate=0.03, colsample_bytree=0.7)
    joblib.dump(xgboost, "models/xgboost.joblib")
    print("XGBoost Regressor model saved as 'models/xgboost.joblib'.")

    # 7. Evaluate their performance on the test set (including cross-validation)
    print("\nEvaluating models on the test set:")

    # Scaled data is used for linear regression, while preprocessed (but not scaled) data is used for the other three models.
    linear_metrics = evaluate_model(linear, X_train_scaled, y_train, X_test_scaled, y_test)
    cross_val_linear = cross_validate_model(linear, X_train_scaled, y_train, cv=5)
    print_evaluation("Linear Regression", linear_metrics)
    print_cross_validation_results("Linear Regression", cross_val_linear)

    decision_tree_metrics = evaluate_model(decision_tree, X_train_processed, y_train, X_test_processed, y_test)
    cross_val_decision_tree = cross_validate_model(decision_tree, X_train_scaled, y_train, cv=5)
    print_evaluation("Decision Tree Regressor", decision_tree_metrics)
    print_cross_validation_results("Decision Tree Regressor", cross_val_decision_tree)

    random_forest_metrics = evaluate_model(random_forest, X_train_processed, y_train, X_test_processed, y_test)
    cross_val_random_forest = cross_validate_model(random_forest, X_train_scaled, y_train, cv=5)
    print_evaluation("Random Forest Regressor", random_forest_metrics)
    print_cross_validation_results("Random Forest Regressor", cross_val_random_forest)

    xgboost_metrics = evaluate_model(xgboost, X_train_processed, y_train, X_test_processed, y_test)
    cross_val_xgboost = cross_validate_model(xgboost, X_train_scaled, y_train, cv=5)
    print_evaluation("XGBoost Regressor", xgboost_metrics)
    print_cross_validation_results("XGBoost Regressor", cross_val_xgboost)


if __name__ == "__main__":
    main()