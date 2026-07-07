import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

NOMINAL_COLUMNS = [
    "type_property", "subtype_property", "heating_type",
    "sun_exposure", "flooding_area_type", "province"
]

PROVINCE_TO_REGION = {
    "brussels": "brussels",
    "vlaams_brabant": "flanders", "antwerp": "flanders", "east_flanders": "flanders",
    "west_flanders": "flanders", "limburg": "flanders",
    "brabant_wallon": "wallonia", "hainaut": "wallonia", "namur": "wallonia",
    "liege": "wallonia", "luxembourg": "wallonia"
}

EPC_TO_KWH_BY_REGION = {
    "brussels": {"A+": -22.5, "A": 22.5, "B": 70.5, "C": 123, "D": 200.5, "E": 295.5, "F": 395.5, "G": 504.5},
    "wallonia": {"A+": -22.5, "A": 65.5, "B": 128, "C": 213, "D": 298, "E": 383, "F": 468, "G": 552},
    "flanders": {"A+": -50, "A": 50, "B": 150.5, "C": 250.5, "D": 350.5, "E": 450.5, "F": 549.5, "G": 650}
}

STATE_OF_PROPERTY_MAPPING = {
    "to_demolish": 0,
    "to_be_renovated": 1, "to_renovate": 1, "to_restore": 1,
    "normal": 2,
    "excellent": 3, "fully_renovated": 3,
    "new": 4, "under_construction": 4
}


class PropertyPreprocessor:
    """
    Fitted on training data once, then reused via joblib both at
    evaluation time in the ML repo and at inference time in the API repo.
    """

    def __init__(self):
        self.onehot_encoders = {}
        self.epc_median = None
        self.state_median = None
        self.train_columns = None

    def _encode_epc_score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        region = df["province"].str.strip().map(PROVINCE_TO_REGION)

        def check_kwh(label, reg):
            if pd.isna(label) or label == "not_specified" or pd.isna(reg):
                return np.nan
            return EPC_TO_KWH_BY_REGION[reg].get(label, np.nan)

        df["epc_score_missing"] = (df["epc_score"] == "not_specified").astype(int)
        df["epc_kwh_m2_year"] = [check_kwh(l, r) for l, r in zip(df["epc_score"], region)]
        return df.drop(columns=["epc_score"])

    def _encode_property_state(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["state_of_property_missing"] = (df["state_of_property"] == "not_specified").astype(int)
        df["state_of_property_encoded"] = df["state_of_property"].map(STATE_OF_PROPERTY_MAPPING)
        return df.drop(columns=["state_of_property"])

    def _onehot_column(self, df: pd.DataFrame, column: str, fit: bool) -> pd.DataFrame:
        if fit:
            encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            encoded = encoder.fit_transform(df[[column]])
            self.onehot_encoders[column] = encoder
        else:
            encoder = self.onehot_encoders[column]
            encoded = encoder.transform(df[[column]])

        encoded_df = pd.DataFrame(
            encoded, columns=encoder.get_feature_names_out([column]), index=df.index
        )
        df = pd.concat([df, encoded_df], axis=1)
        return df.drop(columns=[column])

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._encode_epc_score(df)
        df = self._encode_property_state(df)

        self.epc_median = df["epc_kwh_m2_year"].median()
        df["epc_kwh_m2_year"] = df["epc_kwh_m2_year"].fillna(self.epc_median)

        self.state_median = df["state_of_property_encoded"].median()
        df["state_of_property_encoded"] = df["state_of_property_encoded"].fillna(self.state_median)

        for col in NOMINAL_COLUMNS:
            df = self._onehot_column(df, col, fit=True)

        self.train_columns = list(df.columns)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._encode_epc_score(df)
        df = self._encode_property_state(df)

        df["epc_kwh_m2_year"] = df["epc_kwh_m2_year"].fillna(self.epc_median)
        df["state_of_property_encoded"] = df["state_of_property_encoded"].fillna(self.state_median)

        for col in NOMINAL_COLUMNS:
            df = self._onehot_column(df, col, fit=False)

        return df.reindex(columns=self.train_columns, fill_value=0)