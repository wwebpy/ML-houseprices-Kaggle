from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# =========================
# Pfade & Settings
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "house-prices-advanced-regression-techniques" / "train.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "SalePrice"
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_SPLITS = 5

# =========================
# Helper
# =========================
def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return pre, num_cols, cat_cols


def evaluate(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    # Older scikit-learn versions don't support `squared=False` in mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def cv_rmse(pipe, X, y, n_splits=5, seed=42) -> Tuple[float, float]:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    rmse = np.sqrt(-scores)
    return rmse.mean(), rmse.std()

# =========================
# MODELLE TRAINIEREN
# =========================
def make_models() -> Dict[str, object]:
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=5.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            n_jobs=-1, reg_lambda=1.0
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=31,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    return models


def wrap_log_target(estimator: object, pre: ColumnTransformer) -> TransformedTargetRegressor:
    # Preprocessing in die Pipeline, dann TTR für log1p/exp1m
    return TransformedTargetRegressor(
        regressor=Pipeline([("pre", pre), ("model", estimator)]),
        func=np.log1p, inverse_func=np.expm1
    )


def build_plain_pipe(estimator: object, pre: ColumnTransformer) -> Pipeline:
    return Pipeline([("pre", pre), ("model", estimator)])


def main():
    # ======= Daten =======
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Datendatei nicht gefunden: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Zielspalte '{TARGET}' fehlt.")
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET])

    pre, num_cols, cat_cols = build_preprocessor(X)
    print(f"Daten: {df.shape} | Num: {len(num_cols)} | Cat: {len(cat_cols)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ======= Kandidaten =======
    base_models = make_models()

    # Wir trainieren JEDE Variante 2x:
    #   - raw target (keine Log-Transform)
    #   - log target (TransformedTargetRegressor)
    results = []

    for name, est in base_models.items():
        # ---- RAW ----
        pipe_raw = build_plain_pipe(est, pre)
        print(f"\n🔵 Trainiere {name} (raw target)...")
        pipe_raw.fit(X_train, y_train)
        pred_raw = pipe_raw.predict(X_val)
        m_raw = evaluate(y_val, pred_raw)
        cv_mean_raw, cv_std_raw = cv_rmse(build_plain_pipe(est, pre), X, y, n_splits=N_SPLITS, seed=RANDOM_STATE)
        # speichern
        path_raw = MODELS_DIR / f"{name}_raw.pkl"
        joblib.dump(
            {"pipeline": pipe_raw, "target": TARGET, "use_log_target": False,
             "num_cols": num_cols, "cat_cols": cat_cols, "model_name": name},
            path_raw
        )
        print(f"→ RAW  MAE:{m_raw['MAE']:,.2f}  RMSE:{m_raw['RMSE']:,.2f}  R²:{m_raw['R2']:.3f}  |  CV-RMSE:{cv_mean_raw:,.2f}±{cv_std_raw:,.2f}")
        results.append({
            "Model": name, "Variant": "raw",
            "MAE": m_raw["MAE"], "RMSE": m_raw["RMSE"], "R2": m_raw["R2"],
            "CV_RMSE_mean": cv_mean_raw, "CV_RMSE_std": cv_std_raw,
            "Path": str(path_raw)
        })

        # ---- LOG ----
        pipe_log = wrap_log_target(est, pre)
        print(f"🟣 Trainiere {name} (log target)...")
        pipe_log.fit(X_train, y_train)
        pred_log = pipe_log.predict(X_val)
        m_log = evaluate(y_val, pred_log)
        cv_mean_log, cv_std_log = cv_rmse(wrap_log_target(est, pre), X, y, n_splits=N_SPLITS, seed=RANDOM_STATE)
        # speichern
        path_log = MODELS_DIR / f"{name}_log.pkl"
        joblib.dump(
            {"pipeline": pipe_log, "target": TARGET, "use_log_target": True,
             "num_cols": num_cols, "cat_cols": cat_cols, "model_name": name},
            path_log
        )
        print(f"→ LOG  MAE:{m_log['MAE']:,.2f}  RMSE:{m_log['RMSE']:,.2f}  R²:{m_log['R2']:.3f}  |  CV-RMSE:{cv_mean_log:,.2f}±{cv_std_log:,.2f}")
        results.append({
            "Model": name, "Variant": "log",
            "MAE": m_log["MAE"], "RMSE": m_log["RMSE"], "R2": m_log["R2"],
            "CV_RMSE_mean": cv_mean_log, "CV_RMSE_std": cv_std_log,
            "Path": str(path_log)
        })

    # ======= Ergebnisse sammeln =======
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by=["RMSE", "CV_RMSE_mean"]).reset_index(drop=True)
    print("\n📊 Gesamtergebnis (sortiert nach RMSE):")
    print(df_res[["Model", "Variant", "MAE", "RMSE", "R2", "CV_RMSE_mean", "CV_RMSE_std", "Path"]].round(3))

    # Tabelle speichern
    df_res.to_csv(MODELS_DIR / "results.csv", index=False)
    print(f"\n✅ Ergebnisse gespeichert: {MODELS_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()