# Basic cleaning of data
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Structural cleaning
 
    Steps:
    - drop redundant or not useful columns
    - drop duplicate listings (same property scraped under different IDs)
    - drop rows with price <= 0 or NaN values
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
 
    print(f"clean_data: {n_start} -> {len(df)} rows")
    return df.reset_index(drop=True)