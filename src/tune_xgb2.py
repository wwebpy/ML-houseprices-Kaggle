from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# === Feature Engineering ===
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Cast selected numerical columns to categorical (per request)
    for col in ['MSSubClass', 'MoSold', 'YrSold']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Helper to safely fetch numeric series and fill NaNs for arithmetic
    def _g(name: str, fill=0):
        return df[name].fillna(fill) if name in df.columns else 0

    # Aggregate / domain features
    if {'TotalBsmtSF','1stFlrSF','2ndFlrSF'}.issubset(df.columns):
        df['TotalSF'] = _g('TotalBsmtSF') + _g('1stFlrSF') + _g('2ndFlrSF')
    if {'FullBath','HalfBath','BsmtFullBath','BsmtHalfBath'}.issubset(df.columns):
        df['TotalBath'] = _g('FullBath') + 0.5*_g('HalfBath') + _g('BsmtFullBath') + 0.5*_g('BsmtHalfBath')
    if {'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'}.issubset(df.columns):
        df['TotalPorchSF'] = _g('OpenPorchSF') + _g('EnclosedPorch') + _g('3SsnPorch') + _g('ScreenPorch')
    if {'GarageCars','GarageArea'}.issubset(df.columns):
        df['GarageCapacity'] = _g('GarageCars') * _g('GarageArea')

    # Age features (based on YrSold)
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'].astype('int') - df['YearBuilt']
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df['RemodAge'] = df['YrSold'].astype('int') - df['YearRemodAdd']

    return df

print(f"🔎 Loaded tune_xgb.py from: {__file__}")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "house-prices-advanced-regression-techniques" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_tuned_log.pkl"
TARGET = "SalePrice"
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        print(f"❌ Trainingsdatei fehlt: {DATA_PATH}")
        return

    print("📥 Lade Trainingsdaten …")
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        print(f"❌ Zielspalte '{TARGET}' nicht gefunden. Spalten: {df.columns.tolist()}")
        return

    # Apply feature engineering to training data
    df = build_features(df)

    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET])

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    x_tr, x_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # XGB mit Early Stopping in den Parametern (eval_set wird bei fit übergeben)
    xgb = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=100,
        eval_metric="rmse",
    )

    # --- Debug: verify we run the correct file ---
    print(f"🔎 Running file: {__file__}")

    # Log-Target
    y_tr_log = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)

    # Fit preprocessor first, then transform X into numeric arrays for XGBoost
    print("🧱 Fitte Preprocessing …")
    pre.fit(x_tr)
    X_tr_enc = pre.transform(x_tr)
    X_val_enc = pre.transform(x_val)

    print("🚀 Starte XGBoost-Training (mit Early Stopping) …")
    xgb.fit(
        X_tr_enc, y_tr_log,
        eval_set=[(X_val_enc, y_val_log)],
        verbose=False,
    )

    print("✅ Training fertig. Evaluiere auf Holdout …")
    pred_log = xgb.predict(X_val_enc)
    pred = np.expm1(pred_log)
    rmse = np.sqrt(((pred - y_val) ** 2).mean())
    mae = np.mean(np.abs(pred - y_val))
    print(f"🏁 Holdout – RMSE: {rmse:,.2f} | MAE: {mae:,.2f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "preprocessor": pre,
        "model": xgb,
        "target": TARGET,
        "use_log_target": True,
        "manual_log": True
    }, MODEL_PATH)
    print(f"💾 Modell gespeichert: {MODEL_PATH}")

    # Optional: Submission erzeugen, wenn test.csv existiert
    test_path = DATA_PATH.parent / "test.csv"
    if test_path.exists():
        print("📝 Erzeuge Submission aus getuntem Modell …")
        x_test = pd.read_csv(test_path)
        # Apply the same feature engineering to test data
        x_test = build_features(x_test)
        X_test_enc = pre.transform(x_test)
        preds = np.expm1(xgb.predict(X_test_enc))
        submissions_dir = PROJECT_ROOT / "Submissions"
        submissions_dir.mkdir(exist_ok=True)
        submission_path = submissions_dir / "submission_xgb_tuned.csv"
        pd.DataFrame({"Id": x_test["Id"], "SalePrice": preds}).to_csv(submission_path, index=False)
        print(f"📦 Submission gespeichert: {submission_path}")
    else:
        print(f"ℹ️ Keine Submission erstellt – Testdatei fehlt: {test_path}")


if __name__ == "__main__":
    main()