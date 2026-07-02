import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Structural cleaning
 
    Steps:
    - drop redundant or not useful columns
    - drop duplicate listings (same property scraped under different IDs)
    - drop rows with price <= 0 or NaN values
    - drop top 5% most expensive properties (outlier trimming)
    - drop implausible cheap houses (misclassified properties, different payment structure, etc.)
    """

    df = df.copy()
    n_start = len(df)
 
    # Identifier and redundant columns
    cols_to_drop = {
        "property_id": "Unique identifier without predictive power",
        "city": "Location information to be retreived with lat/long",
        "postal_code": "Location information to be retreived with lat/long"
    }
    cols_present = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_present)
 
    # Duplicate listings: identical on every remaining column
    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"clean_data: dropped {n_dupes} duplicate rows")
 
    # Invalid / missing target
    invalid_price_mask = df["price"].isna() | (df["price"] <= 0)
    print(f"clean_data: dropped {invalid_price_mask.sum()} rows with invalid price")
    df = df.loc[~invalid_price_mask]

    # Drop top 5% most expensive properties (outlier trimming)
    price_cutoff = df["price"].quantile(0.95)
    expensive_mask = df["price"] > price_cutoff
    print(f"clean_data: dropped {expensive_mask.sum()} rows above 95th percentile price ({price_cutoff:,.0f})")
    df = df.loc[~expensive_mask]

    # Drop implausible cheap houses (misclassified properties, different payment structure, etc.)
    cheap_house_mask = (df["type_property"] == "house") & (df["price"] < 40000)
    print(f"clean_data: dropped {cheap_house_mask.sum()} houses priced below 40,000€")
    df = df.loc[~cheap_house_mask]

    #Final result
    print(f"Cleaning complete. {n_start} -> {len(df)} rows")

    return df.reset_index(drop=True)