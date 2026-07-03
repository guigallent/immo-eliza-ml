# Immo Eliza ML
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)


## 📖 Description

**Immo Eliza ML** is a machine learning (ML) pipeline built to predict the price of residential properties in Belgium. The goal of this solo project was to use the `scikit-learn` library in order to **build a complete ML pipeline that goes from raw scraped listings to trained and evaluated regression models.**

This is the third project in the Immo Eliza series. Property listings were first collected in a separate collaborative scraping project, then explored in a collaborative data analysis project (correlations, visualizations, first hypotheses about price drivers), before this project turned those insights into predictive models.

During this project, I:

- Built a robust preprocessing architecture, where encoders and imputers are fit only on training data and reused (never refit) on the test set and at inference time.
- Explored how to convert categorical but complex variables into consistent numerical ones. For example EPC energy labels (which differ in Belgium's Brussels, Flanders, and Wallonia) into estimated kWh/m²/year values, using different conversion tables according to the region.
- Trained and compared four regression models: Linear Regression, Decision Tree, Random Forest, and XGBoost.
- Used cross-validation and `RandomizedSearchCV` to tune hyperparameters.
- Serialized fitted models and preprocessing artifacts with `joblib` to have them ready to be used in the next project (in the repo you will only find the XGBoost file as it is the model that will be used at the next Immo Eliza project).
- As an addition, built an interactive command-line script (`predict.py`) that asks for a property's details and returns a predicted price (to be fine-tuned in the next project).


## 📦 Repo structure

```
immo-eliza-ml/
├── artifacts/
│   └── encoders.joblib
├── data/
│   └── properties_database.csv
├── models/
│   ├── linear.joblib
│   └── xgboost.joblib
├── src/
│   ├── __init__.py
│   ├── clean.py
│   ├── config.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── train.py
├── .gitignore
├── main.py
├── predict.py
├── README.md
└── requirements.txt
```

### 🧩 Project modules

- `src/clean.py` prepares the raw scraped data by dropping redundant columns, removing duplicates, invalid or implausible prices, and outliers.
- `src/preprocess.py` handles feature engineering and encoding (EPC score conversion, property state ordinal encoding, and one-hot encoding of categorical variables).
- `src/train.py` performs the train/test split, feature scaling, and training of all four regression models.
- `src/evaluate.py` computes evaluation metrics (MAE, RMSE, R², MAPE, etc.) and cross-validation scores for each model.
- `src/config.py` centralizes `RANDOM_STATE` to avoid circular imports between modules.
- `predict.py` loads the trained model and fitted preprocessing artifacts, and predicts the price of a new property according to the user's input.


## 🔀 Pipeline overview

The pipeline is executed through `main.py`:

```
Raw scraped data (properties_database.csv)
        |
        v
clean_data() → drop irrelevant columns, duplicates, invalid prices, outliers
        |
        v
split_data() → train / test split (80/20)
        |
        v
preprocess_data() → EPC & property state encoding, one-hot encoding
        |
        v
scale_data() → StandardScaler (fit on train, transform test)
        |
        v
Train models → Linear Regression, Decision Tree, Random Forest, XGBoost
        |
        v
Save models & preprocessing artifacts (joblib) → models/ , artifacts/
        |
        v
Evaluate on test set → MAE, RMSE, R², cross-validation
```

Once artifacts are saved, `predict.py` can reload them and generate predictions for new properties.


## 📌 Usage

1. Clone the repository to your local machine.

2. Create and activate your virtual environment.

3. Install the libraries listed in the `requirements.txt` file.

4. Run `main.py` to clean the data, train the models, and save the trained models and preprocessing artifacts.

5. Run `predict.py` to get a price prediction for a new property.


## 📊 Results & model choices

To evaluate the impact of data cleaning on model performance, I compared results from an early iteration (before removing price outliers, implausible sub-€40k house listings, and some columns that did not contribute much to determining price) against the final, more thoroughly cleaned dataset. Here is a simplified table with the results:

| Model | Stage | MAE | RMSE | MAPE | R² Test | R² Gap | CV Mean R² | CV Std |
|---|---|---|---|---|---|---|---|---|
| **Linear Regression** | Before | €123,996.22 | €231,689.87 | 34.85% | 0.5341 | 0.0258 | 0.5450 | 0.0350 |
| | After | €80,432.44 | €106,908.26 | 28.61% | 0.5708 | 0.0286 | 0.5888 | 0.0119 |
| **Decision Tree** | Before | €123,829.51 | €236,881.19 | 33.98% | 0.5130 | 0.1552 | 0.5250 | 0.0653 |
| | After | €80,837.61 | €109,696.74 | 27.80% | 0.5481 | 0.1345 | 0.5509 | 0.0237 |
| **Random Forest** | Before | €96,457.65 | €193,563.51 | 28.03% | 0.6748 | 0.2809 | 0.6717 | 0.0405 |
| | After | €65,010.55 | €90,378.88 | 23.31% | 0.6933 | 0.2633 | 0.6807 | 0.0152 |
| **XGBoost** | Before | €86,380.68 | €174,088.48 | 23.63% | 0.7370 | 0.2360 | 0.7109 | 0.0500 |
| | **After** | **€58,079.14** | **€82,635.61** | **19.89%** | **0.7436** | 0.2079 | **0.7332** | **0.0191** |

Cleaning the dataset more thoroughly clearly improved every model. This is most visible in RMSE, which roughly halved across the board—a result of both model improvement and the mathematical effect of removing price outliers that previously distorted the squared error metrics. In addition, cross-validation stability improved significantly, with standard deviations dropping by 2–3 times. This indicates that the earlier iteration's high CV variance was driven by noisy rows randomly disrupting specific folds, rather than consistent model performance.

**XGBoost was the best-performing model** in both iterations and was selected as the model to deploy in the next Immo Eliza project. It combines the lowest error metrics (MAE, RMSE, MAPE) with the highest R² among all four models tested and a strong cross-validation performance.

The R² Gap column, however, may seem counterintuitive. Linear Regression has the smallest gap (0.0258) because it is too simple to overfit and not because it is the best model (its R² of 0.5708 is the weakest of the four). Decision Tree, on the other hand, has a smaller gap than XGBoost (0.1345 vs. 0.2079), which it may appear as less overfitting. However, its test R² (0.5481) is actually the *lowest* of all four models. This implies that it is not generalizing better, but rather not learning much in either the train or test set.

XGBoost's gap (0.2079) means it is still overfitting, even after tuning. This, in fact, is a common issue when using XGBoost models and dealing with it would require further fine-tuning of its hyperparameters. What justifies choosing it anyway is that its test-set performance is the highest by a clear margin, *and* its cross-validation scores are both the highest (0.7332) and the most stable (std 0.0191).

Random Forest is the most consistent model across folds, with the lowest CV standard deviation (0.0152) of the four. Its test R² (0.6933) is also the second-best of the group. However, it is the clearest overfitting case of the four, as its gap (0.2633) is the largest, and still underperforms XGBoost on every test metric.


## 🔧 Possible improvements

While the pipeline works end-to-end, there are a few ways it could be improved in the future:

### 1. Use a scikit-learn `Pipeline`

Right now, preprocessing and model training are handled by separate, manually-sequenced functions. Using scikit-learn's `Pipeline` would bring a few concrete benefits:

- **Cleaner code**: `main.py` and `predict.py` would shrink to a `pipeline.fit(...)` / `pipeline.predict(...)` call each, instead of threading intermediate DataFrames through several functions.
- **Simpler serialization**: the entire preprocessing + model chain can be saved as a single `joblib` file, instead of separately saving encoders, scaler, and model and having to re-apply them in the right order in `predict.py`.
- **Less risk of leakage by construction**: a `Pipeline` guarantees that `fit_transform` only happens on training folds during cross-validation and `GridSearchCV`/`RandomizedSearchCV`, rather than relying on manually passing them.

### 2. Add postal code back as a feature

`postal_code` was dropped early in this project in favor of latitude/longitude to simplify the encoding. However, it can be very useful in the next Immo Eliza project for the user to provide geographical information more intuitively.

### 3. A more robust `predict.py`

The current `predict.py` serves as a first attempt to check how the XGBoost model predicts prices. This will be overhauled as part of the next stage of the Immo Eliza pipeline.


## ⏱️ Timeline

This project took five days for completion.


## 📌 Personal Situation

This project was done as part of the AI & Data Science Bootcamp at BeCode, as a solo project. It is the third stage of the Immo Eliza pipeline, following a property-scraping project and a data-analysis/visualization project, both available on my profile.

👥 Connect with me:
- [LinkedIn - Guillermo Gallent Lloria](https://www.linkedin.com/in/guillermo-gallent/)
