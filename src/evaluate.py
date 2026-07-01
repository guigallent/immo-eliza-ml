import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def evaluate_model(model, X_test_scaled: np.ndarray, y_test: pd.Series) -> dict:
    """
    Evaluates the trained model on the test set.
    """
    y_pred = model.predict(X_test_scaled)
    
    return {
        "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_pred),
        "Root Mean Squared Error (RMSE)": root_mean_squared_error(y_test, y_pred),
        "R² Score": r2_score(y_test, y_pred)
    }