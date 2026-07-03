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
    - scaler: Fitted StandardScaler, for use on new data at inference time.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    return X_train_scaled, X_test_scaled, scaler

def linear_regression(X_train_scaled: np.ndarray, y_train: pd.Series):
    """Trains a regression model on the scaled training data."""

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print("\nLinear Regression model trained.")
    return model


def decision_tree_regression(X_train_scaled: np.ndarray, y_train: pd.Series, min_samples_split: int = 40, min_samples_leaf: int = 16, max_depth: int = 20, ccp_alpha: float = 0.1):
    """
    Trains a Decision Tree Regressor on the scaled training data.
    
    Parameters:
    - X_train_scaled: Scaled training features.
    - y_train: Training target variable (price).
    - max_depth: The maximum depth of the tree.
    - min_samples_split: The minimum number of samples required to split an internal node.
    - min_samples_leaf: The minimum number of samples required to be at a leaf node.
    - ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning.
    
    Returns:
    - model: Trained Decision Tree Regressor model.
    """

    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    print(f"\nDecision Tree Regressor trained with max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}.")
    return model


def random_forest_regression(X_train_scaled: np.ndarray, y_train: pd.Series, n_estimators: int = 300, min_samples_split: int = 2, min_samples_leaf: int = 1, max_features: str = 'sqrt', max_depth: int = 30):
    """
    Trains a Random Forest Regressor on the scaled training data.
    
    Parameters:
    - X_train_scaled: Scaled training features.
    - y_train: Training target variable (price).
    - n_estimators: The number of trees in the forest.
    - min_samples_split: The minimum number of samples required to split an internal node.
    - min_samples_leaf: The minimum number of samples required to be at a leaf node.
    - max_features: The number of features to consider when looking for the best split.
    - max_depth: The maximum depth of the tree.
    
    Returns:
    - model: Trained Random Forest Regressor model.
    """

    model = RandomForestRegressor(n_estimators = n_estimators, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, max_features = max_features, max_depth = max_depth, random_state = RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    print(f"\nRandom Forest Regressor trained with n_estimators={n_estimators}.")
    return model

def xgboost_regression(X_train_scaled: np.ndarray, y_train: pd.Series, subsample: float = 0.9, reg_lambda: float = 10, reg_alpha: float = 0, n_estimators: int = 1200, min_child_weight: int = 1, max_depth: int = 7, learning_rate: float = 0.03, colsample_bytree: float = 0.7):
    """
    Trains an XGBoost Regressor on the scaled training data.
    
    Parameters:
    - X_train_scaled: Scaled training features.
    - y_train: Training target variable (price).
    - subsample: The fraction of observations to be used for fitting each tree.
    - reg_lambda: L2 regularization term on weights.
    - reg_alpha: L1 regularization term on weights.
    - n_estimators: The number of trees in the forest.
    - min_child_weight: Minimum sum of instance weight needed in a child.
    - max_depth: The maximum depth of the tree.
    - learning_rate: Step size shrinkage used in update to prevents overfitting.
    - colsample_bytree: Fraction of columns to be considered for each split.

    Returns:
    - model: Trained XGBoost Regressor model.
    """
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)

    print(f"\nXGBoost Regressor trained with n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}.")
    return model