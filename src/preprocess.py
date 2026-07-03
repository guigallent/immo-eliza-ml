import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# List of nominal columns to be one-hot encoded. 
# Keeping this list here allows for easy modification in the future if needed.
NOMINAL_COLUMNS = [
    "type_property",
    "subtype_property",
    "heating_type",
    "sun_exposure",
    "flooding_area_type",
    "province"
]

def encode_epc_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding of EPC score. Each province's code system is different. 
    Here, the system takes the medium value of each label. 
    The info on the labels was extracted from official soruces.
    """

    """
    Some extra information on the EPC score and its meaning in each Belgian region:
    Brussels
    Label A: ≤ 45 kWh/m²/year
    Label B: 46–95 kWh/m²/year
    Label C: 96–150 kWh/m²/year
    Label D: 151–250 kWh/m²/year
    Label E: 251–340 kWh/m²/year
    Label F: 341–450 kWh/m²/year
    Label G: > 450 kWh/m²/year
    https://www.certinergie.be/en/energy-performance-certificate/epc-brussels/
    
    Wallonia
    Label A++: < 0 kWh/m²/year
    Label A: ≤ 45 kWh/m²/year
    Label A: ≤ 85 kWh/m²/year
    Label B: 86–170 kWh/m²/year
    Label C: 171–255 kWh/m²/year
    Label D: 256–340 kWh/m²/year
    Label E: 341–425 kWh/m²/year
    Label F: 426–510 kWh/m²/year
    Label G: > 510 kWh/m²/year
    https://www.certinergie.be/en/energy-performance-certificate/epc-certificate-wallonia/
 
    Flanders
    Label A+: < 0 kWh/m²/year
    Label A: 0-100 kWh/m²/year
    Label B: 101-200 kWh/m²/year
    Label C: 201-300 kWh/m²/year
    Label D: 301-400 kWh/m²/year
    Label E: 401-500 kWh/m²/year
    Label F: > 500 kWh/m²/year
    https://assets.vlaanderen.be/image/upload/v1706105167/VoorbeeldEPCvanafjanuari2019_nieuw_sg5wa0.pdf 
    """

    #Identify each province with their region
    province_to_region ={ "brussels": "brussels", 
                         
                         "vlaams_brabant": "flanders", "antwerp": "flanders", "east_flanders": "flanders", 
                         "west_flanders": "flanders", "limburg": "flanders",

                         "brabant_wallon": "wallonia", "hainaut": "wallonia", "namur": "wallonia", 
                         "liege": "wallonia", "luxembourg": "wallonia"
    }

    # Use the middle value for each label according to official data 
    epc_to_kwh_by_region = {
        "brussels": {"A+": -22.5, "A": 22.5, "B": 70.5, "C": 123, "D": 200.5, "E": 295.5, "F": 395.5, "G": 504.5},
        "wallonia": {"A+": -22.5, "A": 65.5, "B": 128, "C": 213, "D": 298, "E": 383, "F": 468, "G": 552},
        "flanders": {"A+": -50, "A": 50, "B": 150.5, "C": 250.5, "D": 350.5, "E": 450.5, "F": 549.5, "G": 650}
    }

    df = df.copy()
    region = df["province"].str.strip().map(province_to_region)
    
    def check_kwh(label: str, reg: str) -> float:
        if pd.isna(label) or label == "not_specified" or pd.isna(reg):
            return np.nan
        return epc_to_kwh_by_region[reg].get(label, np.nan)
    
    df["epc_score_missing"] = (df["epc_score"] == "not_specified").astype(int)
    df["epc_kwh_m2_year"] = [
        check_kwh(label, reg) for label, reg in zip(df["epc_score"], region)
    ]
    df = df.drop(columns=["epc_score"])
    return df

def encode_property_state(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal encoding of property condition (0=worst to 4=best)."""

    state_of_property_mapping = {
        "to_demolish": 0,
        "to_be_renovated": 1, "to_renovate": 1, "to_restore": 1,
        "normal": 2,
        "excellent": 3, "fully_renovated": 3,
        "new": 4, "under_construction": 4
    }

    df = df.copy()
    df["state_of_property_missing"] = (df["state_of_property"] == "not_specified").astype(int)
    df["state_of_property_encoded"] = df["state_of_property"].map(state_of_property_mapping)
    df = df.drop(columns = ["state_of_property"])
    return df

def encode_onehot(df: pd.DataFrame, column: str, encoder: OneHotEncoder = None) -> tuple[pd.DataFrame, OneHotEncoder]:
    """One-hot encoding of a categorical column. If an encoder is provided, it will be used to transform the data; otherwise, a new encoder will be fitted."""
    
    if encoder is None:
        encoder = OneHotEncoder(drop = "first", sparse_output = False, handle_unknown = "ignore")
        encoded = encoder.fit_transform(df[[column]])
    else:
        encoded = encoder.transform(df[[column]])

    encoded_df = pd.DataFrame(
        encoded,
        columns = encoder.get_feature_names_out([column]),
        index = df.index,
    )

    df = pd.concat([df, encoded_df], axis = 1)
    df = df.drop(columns = [column])
    return df, encoder

def preprocess_data(df: pd.DataFrame, encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """It applies the preprocessing steps to the DataFrame."""

    df = encode_epc_score(df)
    df = encode_property_state(df)

    is_training = encoders is None
    if is_training:
        encoders = {}

    # --- Impute EPC kWh/m² ---
    if is_training:
        epc_median = df["epc_kwh_m2_year"].median()
        encoders["epc_median"] = epc_median
    else:
        epc_median = encoders["epc_median"]
    df["epc_kwh_m2_year"] = df["epc_kwh_m2_year"].fillna(epc_median)

    # --- Impute state_of_property_encoded ---
    if is_training:
        state_median = df["state_of_property_encoded"].median()
        encoders["state_median"] = state_median
    else:
        state_median = encoders["state_median"]
    df["state_of_property_encoded"] = df["state_of_property_encoded"].fillna(state_median)

    # --- One-hot encode nominal columns ---
    for col in NOMINAL_COLUMNS:
        df, fitted_encoder = encode_onehot(df, col, encoder=encoders.get(col))
        encoders[col] = fitted_encoder

    return df, encoders

