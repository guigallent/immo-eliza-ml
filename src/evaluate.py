import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, cross_val_score
from src.config import RANDOM_STATE

def evaluate_model(model, X_train_scaled: np.ndarray, y_train: pd.Series, X_test_scaled: np.ndarray, y_test: pd.Series) -> dict:
    """Evaluates the trained model on the test set."""
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    return {
        "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_pred_test),
        "Median Absolute Error": median_absolute_error(y_test, y_pred_test),
        "Root Mean Squared Error (RMSE)": root_mean_squared_error(y_test, y_pred_test),
        "Mean Absolute Percentage Error (MAPE)": mean_absolute_percentage_error(y_test, y_pred_test),
        "R² Score (Test)": r2_score(y_test, y_pred_test),
        "R² Score (Train)": r2_score(y_train, y_pred_train),
        "R² Gap (Train - Test)": r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test)
    }

def print_evaluation(model_name: str, metrics: dict) -> None:
    """Prints evaluation metrics for a single model in a pretty way."""

    print(f"\n{'='*50}")
    print(f"{model_name}")
    print(f"{'='*50}")
    for metric_name, value in metrics.items():
        if "Percentage" in metric_name:
            print(f"{metric_name:<35} {value:.2%}")
        elif "R²" in metric_name:
            print(f"{metric_name:<35} {value:.4f}")
        else:
            print(f"{metric_name:<35} {value:,.2f}")

def cross_validate_model(model, X: np.ndarray, y: pd.Series, cv: int = 5) -> dict:
    """Performs cross-validation on the model and returns the mean and standard deviation of R² scores."""

    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    
    return {
        "R² Scores": scores,
        "Mean R²": scores.mean(),
        "Std R²": scores.std()
    }

def print_cross_validation_results(model_name: str, cv_results: dict) -> None:
    """Prints cross-validation results for a single model in a pretty way."""

    print(f"\n{'='*50}")
    print(f"{model_name} - Cross-Validation Results")
    print(f"{'='*50}")
    print(f"R² Scores: {cv_results['R² Scores']}")
    print(f"Mean R²: {cv_results['Mean R²']:.4f}")
    print(f"Std R²: {cv_results['Std R²']:.4f}")