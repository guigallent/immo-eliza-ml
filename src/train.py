import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    
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
    - scaler: Fitted StandardScaler, for use on new data at inference time.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    return X_train_scaled, X_test_scaled, scaler

def linear_regression(X_train_scaled: np.ndarray, y_train: pd.Series):
    """
    Trains a regression model on the scaled training data.
    """
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print("\nLinear Regression model trained.")
    return model


def decision_tree_regression(X_train_scaled: np.ndarray, y_train: pd.Series, max_depth: int = 3):
    """
    Trains a Decision Tree Regressor on the scaled training data.
    
    Parameters:
    - X_train_scaled: Scaled training features.
    - y_train: Training target variable (price).
    - max_depth: The maximum depth of the tree.
    
    Returns:
    - model: Trained Decision Tree Regressor model.
    """
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    print(f"\nDecision Tree Regressor trained with max_depth={max_depth}.")
    return model


def random_forest_regression(X_train_scaled: np.ndarray, y_train: pd.Series, n_estimators: int = 100):
    """
    Trains a Random Forest Regressor on the scaled training data.
    
    Parameters:
    - X_train_scaled: Scaled training features.
    - y_train: Training target variable (price).
    - n_estimators: The number of trees in the forest.
    
    Returns:
    - model: Trained Random Forest Regressor model.
    """

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    print(f"\nRandom Forest Regressor trained with n_estimators={n_estimators}.")
    return model

def xgboost_regression(X_train_scaled: np.ndarray, y_train: pd.Series, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.05):
    """
    Trains an XGBoost Regressor on the scaled training data.
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)

    print(f"\nXGBoost Regressor trained with n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}.")
    return model