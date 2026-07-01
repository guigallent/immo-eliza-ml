import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from main import RANDOM_STATE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets.
    
    Parameters:
    - df: The cleaned DataFrame to split.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Seed used by the random number generator for reproducibility.
    
    Returns:
    - X_train: Training features.
    - X_test: Testing features.
    - y_train: Training target variable (price).
    - y_test: Testing target variable (price).
    """
    X = df.drop("price", axis=1)
    y = df["price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def scale_data(X_train_processed: pd.DataFrame, X_test_processed: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Scales the training and testing feature sets using StandardScaler.
    
    Parameters:
    - X_train_processed: Preprocessed training features.
    - X_test_processed: Preprocessed testing features.
    
    Returns:
    - X_train_scaled: Scaled training features.
    - X_test_scaled: Scaled testing features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    return X_train_scaled, X_test_scaled

def train_linear_regression(X_train_scaled: np.ndarray, y_train: pd.Series):
    """
    Trains a regression model on the scaled training data.
    """
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model


def evaluate_model(model, X_test_scaled: np.ndarray, y_test: pd.Series) -> dict:
    """
    Evaluates the trained model on the test set.
    """
    y_pred = model.predict(X_test_scaled)
    
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }