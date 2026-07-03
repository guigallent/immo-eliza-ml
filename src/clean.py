import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Structural cleaning
 
    Steps:
    - drop redundant or not useful columns
    - drop duplicate listings (same property scraped under different IDs)
    - drop rows with price <= 0 or NaN values
    - drop implausible cheap houses (misclassified properties, different payment structure, etc.)
    - drop top 5% most expensive properties (outlier trimming)
    """

    df = df.copy()
    n_start = len(df)
 
    # Identifier and redundant columns
    cols_to_drop = {
        "property_id": "Unique identifier without predictive power",
        "city": "Location information to be retreived with lat/long",
        "postal_code": "Location information to be retreived with lat/long",
        "distance_from_train_stations_by_foot": "Not that relevant for price prediction",
        "distance_from_elementary_school_by_foot": "Not that relevant for price prediction",
        "distance_from_high_school_by_foot": "Not that relevant for price prediction"
    }
    cols_present = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_present)
 
    # Duplicate listings: identical on every remaining column
    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"clean_data: dropped {n_dupes} duplicate rows")
 
    # Invalid / missing price
    invalid_price_mask = df["price"].isna() | (df["price"] <= 0)
    print(f"clean_data: dropped {invalid_price_mask.sum()} rows with invalid price")
    df = df.loc[~invalid_price_mask]

    # Drop implausible cheap houses (misclassified properties, different payment structure, etc.)
    # Runs before the percentile trim so these junk rows don't skew the cutoff.
    cheap_house_mask = (df["type_property"] == "house") & (df["price"] < 40000)
    print(f"clean_data: dropped {cheap_house_mask.sum()} houses priced below 40,000€")
    df = df.loc[~cheap_house_mask]

    # Drop top 5% most expensive properties (outlier trimming)
    price_cutoff = df["price"].quantile(0.95)
    expensive_mask = df["price"] > price_cutoff
    print(f"clean_data: dropped {expensive_mask.sum()} rows above 95th percentile price ({price_cutoff:,.0f})")
    df = df.loc[~expensive_mask]

    #Final result
    print(f"Cleaning complete. {n_start} -> {len(df)} rows")

    return df.reset_index(drop=True)