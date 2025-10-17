import pandas as pd
import numpy as np
from scipy.stats import skew

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Wichtige numerische → kategorische Casts
    for col in ["MSSubClass", "MoSold", "YrSold"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Helper (Null-sichere Addition)
    def _g(name: str, fill=0):
        return df[name].fillna(fill) if name in df.columns else 0

    # 2) Kern-Features
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = _g("TotalBsmtSF") + _g("1stFlrSF") + _g("2ndFlrSF")

    if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(df.columns):
        df["TotalBath"] = (
            _g("FullBath") + 0.5 * _g("HalfBath") +
            _g("BsmtFullBath") + 0.5 * _g("BsmtHalfBath")
        )

    if {"OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"}.issubset(df.columns):
        df["TotalPorchSF"] = (
            _g("OpenPorchSF") + _g("EnclosedPorch") +
            _g("3SsnPorch") + _g("ScreenPorch")
        )

    if {"GarageCars", "GarageArea"}.issubset(df.columns):
        df["GarageCapacity"] = _g("GarageCars") * _g("GarageArea")

    # 3) Alters-Features
    if "YrSold" in df.columns and "YearBuilt" in df.columns:
        df["HouseAge"] = df["YrSold"].astype(int) - df["YearBuilt"]
    if "YrSold" in df.columns and "YearRemodAdd" in df.columns:
        df["RemodAge"] = df["YrSold"].astype(int) - df["YearRemodAdd"]

    # 4) Sinnvolle Imputationen
    # LotFrontage nach Neighborhood-Median
    if {"Neighborhood", "LotFrontage"}.issubset(df.columns):
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"] \
                              .transform(lambda s: s.fillna(s.median()))

    # 5) Rare-Label-Bündelung (reduziert Rauschen)
    def rare_to_other(series: pd.Series, min_freq: int = 20) -> pd.Series:
        vc = series.value_counts(dropna=False)
        rare = vc[vc < min_freq].index
        return series.where(~series.isin(rare), other="Other")

    for col in ["Neighborhood", "Exterior1st", "Exterior2nd", "Condition1", "Condition2",
                "RoofMatl", "MasVnrType", "Electrical", "SaleType", "SaleCondition"]:
        if col in df.columns and df[col].dtype.name in ("object", "category"):
            df[col] = rare_to_other(df[col])

    # 6) Interaktionen / Verhältnisse
    if {"OverallQual", "TotalSF"}.issubset(df.columns):
        df["Qual_SF"] = df["OverallQual"] * df["TotalSF"]

    if {"TotalSF", "TotRmsAbvGrd"}.issubset(df.columns):
        df["SF_perRoom"] = df["TotalSF"] / df["TotRmsAbvGrd"].replace(0, np.nan)

    if {"TotalBath", "BedroomAbvGr"}.issubset(df.columns):
        df["BathPerBed"] = df["TotalBath"] / df["BedroomAbvGr"].replace(0, np.nan)

    if {"OverallQual", "OverallCond"}.issubset(df.columns):
        df["QualCond"] = df["OverallQual"] * df["OverallCond"]

    # Verhältnisse: NaN/Inf abfangen
    for col in ["SF_perRoom", "BathPerBed"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df