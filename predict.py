import pandas as pd
import joblib
from preprocess import preprocess_data

MODEL_PATH = "models/xgboost.joblib"
ENCODERS_PATH = "artifacts/encoders.joblib"

COLS_TO_DROP = ["property_id", "city", "postal_code"]

VALID_CATEGORIES = {
    "province": ["antwerp", "brabant_wallon", "brussels", "east_flanders", "hainaut",
                 "liege", "limburg", "luxembourg", "namur", "vlaams_brabant", "west_flanders"],
    "type_property": ["apartment", "house"],
    "subtype_property": ["apartment", "bungalow", "chalet", "cottage", "duplex", "ground_floor",
                          "loft", "mansion", "master_house", "mixed_building", "penthouse",
                          "residence", "studio", "triplex", "villa"],
    "state_of_property": ["excellent", "fully_renovated", "new", "normal", "not_specified",
                           "to_be_renovated", "to_demolish", "to_renovate", "to_restore",
                           "under_construction"],
    "heating_type": ["coal", "electricity", "fuel_oil", "gas", "hot_air", "not_specified",
                      "solar_energy", "wood"],
    "sun_exposure": ["east", "north", "north_east", "north_west", "not_specified", "south",
                      "south_east", "south_west", "west"],
    "epc_score": ["A", "A+", "B", "C", "D", "E", "F", "G", "not_specified"],
    "flooding_area_type": ["(information_not_available)", "actual_flooding_area", "low_risk",
                            "no_flooding_area", "possible_flooding_area"],
}

BINARY_FIELDS = ["terrace", "garden", "garage", "swimming_pool"]

INT_FIELDS = ["facades", "bedrooms", "bathrooms", "toilets"]

FLOAT_FIELDS = ["livable_surface", "latitude", "longitude",
                 "distance_from_train_stations_by_foot",
                 "distance_from_elementary_school_by_foot",
                 "distance_from_high_school_by_foot"]


def load_artifacts(model_path: str = MODEL_PATH, encoders_path: str = ENCODERS_PATH):
    """Loads the trained model and fitted preprocessing encoders."""

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    return model, encoders


def prepare_input(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Applies the same structural cleaning and preprocessing used in training,
    using already-fitted encoders. No fitting happens here.
    """

    df = df.copy()

    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present)

    df_processed, _ = preprocess_data(df, encoders=encoders)

    # Align to the exact column set/order seen during training.
    # Missing columns are filled with 0.
    # Any unexpected extra columns are dropped.
    train_columns = encoders["train_columns"]
    df_processed = df_processed.reindex(columns=train_columns, fill_value=0)

    return df_processed


def predict(df: pd.DataFrame) -> pd.Series:
    """
    Runs inference on new, raw property data.

    Parameters:
    - df: raw DataFrame with the same input columns as the original scraped data, excluding "price" (that's what we're predicting).

    Returns:
    - pd.Series of predicted prices, indexed like the input.
    """

    model, encoders = load_artifacts()
    X = prepare_input(df, encoders)
    predictions = model.predict(X)

    return pd.Series(predictions, index=df.index, name="predicted_price")


def _prompt_choice(field: str) -> str:
    """Prompts for a categorical field, validating against known training values (case-insensitive)."""

    options = VALID_CATEGORIES[field]
    options_str = ", ".join(options)
    lookup = {opt.lower(): opt for opt in options} 
    while True:
        raw = input(f"{field} ({options_str}): ").strip()
        match = lookup.get(raw.lower())
        if match is not None:
            return match
        print(f"  '{raw}' is not a recognized value for {field}. Please choose from the list above.")


def _prompt_float(field: str) -> float:
    """Prompts for a numeric (float) field."""

    while True:
        raw = input(f"{field}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print(f"  '{raw}' is not a valid number. Please try again.")


def _prompt_int(field: str) -> int:
    """Prompts for a numeric (integer) field."""

    while True:
        raw = input(f"{field}: ").strip()
        try:
            return int(raw)
        except ValueError:
            print(f"  '{raw}' is not a valid whole number. Please try again.")


def _prompt_binary(field: str) -> int:
    """Prompts for a yes/no field, stored as 0/1 to match training data."""

    while True:
        raw = input(f"{field} (yes/no): ").strip().lower()
        if raw in ("yes", "y", "1"):
            return 1
        if raw in ("no", "n", "0"):
            return 0
        print("  Please answer yes or no.")


def prompt_new_property() -> pd.DataFrame:
    """Interactively collects raw property details from the user and returns a one-row DataFrame."""

    print("Enter the details of the property you'd like a price prediction for.\n")

    data = {}
    data["province"] = _prompt_choice("province")
    data["type_property"] = _prompt_choice("type_property")
    data["subtype_property"] = _prompt_choice("subtype_property")
    data["state_of_property"] = _prompt_choice("state_of_property")
    data["heating_type"] = _prompt_choice("heating_type")
    data["sun_exposure"] = _prompt_choice("sun_exposure")
    data["epc_score"] = _prompt_choice("epc_score")
    data["flooding_area_type"] = _prompt_choice("flooding_area_type")

    data["livable_surface"] = _prompt_float("livable_surface (m²)")
    data["latitude"] = _prompt_float("latitude")
    data["longitude"] = _prompt_float("longitude")
    data["distance_from_train_stations_by_foot"] = _prompt_float("distance_from_train_stations_by_foot (meters)")
    data["distance_from_elementary_school_by_foot"] = _prompt_float("distance_from_elementary_school_by_foot (meters)")
    data["distance_from_high_school_by_foot"] = _prompt_float("distance_from_high_school_by_foot (meters)")

    data["facades"] = _prompt_int("facades")
    data["bedrooms"] = _prompt_int("bedrooms")
    data["bathrooms"] = _prompt_int("bathrooms")
    data["toilets"] = _prompt_int("toilets")

    for field in BINARY_FIELDS:
        data[field] = _prompt_binary(field)

    return pd.DataFrame([data])


if __name__ == "__main__":
    new_property = prompt_new_property()
    predicted_price = predict(new_property)
    print(f"\nPredicted price: €{predicted_price.iloc[0]:,.2f}")