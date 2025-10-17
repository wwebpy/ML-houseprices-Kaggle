from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
import xgboost as xgb

# === Feature Engineering ===
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Gewünschte kategorische Casts
    for col in ["MSSubClass", "MoSold", "YrSold"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Safe getter
    def _g(name: str, fill=0):
        return df[name].fillna(fill) if name in df.columns else 0

    # Domain-Features
    if {"TotalBsmtSF","1stFlrSF","2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = _g("TotalBsmtSF") + _g("1stFlrSF") + _g("2ndFlrSF")
    if {"FullBath","HalfBath","BsmtFullBath","BsmtHalfBath"}.issubset(df.columns):
        df["TotalBath"] = _g("FullBath") + 0.5*_g("HalfBath") + _g("BsmtFullBath") + 0.5*_g("BsmtHalfBath")
    if {"OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"}.issubset(df.columns):
        df["TotalPorchSF"] = _g("OpenPorchSF") + _g("EnclosedPorch") + _g("3SsnPorch") + _g("ScreenPorch")
    if {"GarageCars","GarageArea"}.issubset(df.columns):
        df["GarageCapacity"] = _g("GarageCars") * _g("GarageArea")

    # Altersfeatures
    if "YrSold" in df.columns and "YearBuilt" in df.columns:
        df["HouseAge"] = df["YrSold"].astype(int) - df["YearBuilt"]
    if "YrSold" in df.columns and "YearRemodAdd" in df.columns:
        df["RemodAge"] = df["YrSold"].astype(int) - df["YearRemodAdd"]

    # Kleine Interaktion (oft nützlich)
    if {"OverallQual", "TotalSF"}.issubset(df.columns):
        df["Qual_SF"] = df["OverallQual"] * df["TotalSF"]

    return df


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
        print(f"❌ Zielspalte '{TARGET}' nicht gefunden.")
        return

    # Ziel vor dem Capping auf float casten (vermeidet dtype-Warnungen)
    df[TARGET] = df[TARGET].astype(float)

    # === Feature Engineering ===
    df = build_features(df)

    # Optional: leichte Outlier-Begrenzung (wirkt stabilisierend)
    q_hi = df[TARGET].quantile(0.995)
    df.loc[df[TARGET] > q_hi, TARGET] = q_hi

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
    y_tr_log, y_val_log = np.log1p(y_tr), np.log1p(y_val)

    print("🧱 Fitte Preprocessing …")
    pre.fit(x_tr)
    X_tr_enc = pre.transform(x_tr)
    X_val_enc = pre.transform(x_val)

    # === Randomized Search (CV auf log-Target, ohne Early Stopping) ===
    param_dist = {
        "n_estimators": [1000, 1500, 2000, 3000],
        "learning_rate": [0.03, 0.04, 0.05, 0.06, 0.08],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 2, 3, 5, 8, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "gamma": [0, 0.5, 1, 2, 3],
        "reg_alpha": [0.0, 0.001, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 3.0, 5.0, 8.0, 10.0],
    }

    print("🔍 Starte RandomizedSearchCV (5-fold CV, log target) …")
    base_xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=40,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    search.fit(X_tr_enc, y_tr_log)
    best_params = search.best_params_

    print(f"✅ Beste CV-Params: {best_params}")
    print(f"✅ Beste CV-RMSE: {-search.best_score_:,.2f}")

    # === Retrain mit Early Stopping (Callbacks, kein eval_metric in fit/ctor) ===
    print("🚀 Retrain best params mit Early Stopping …")
    xgb_best = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    # Version-adaptive early stopping: try callbacks (xgboost>=2.0), fallback to early_stopping_rounds (xgboost<2.0)
    try:
        xgb_best.fit(
            X_tr_enc, y_tr_log,
            eval_set=[(X_val_enc, y_val_log)],
            callbacks=[xgb.callback.EarlyStopping(rounds=100, save_best=True, maximize=False)]
        )
    except TypeError:
        try:
            xgb_best.fit(
                X_tr_enc, y_tr_log,
                eval_set=[(X_val_enc, y_val_log)],
                early_stopping_rounds=100,
                verbose=False
            )
        except TypeError:
            # As a last resort, fit without early stopping
            xgb_best.fit(X_tr_enc, y_tr_log)

    # Holdout-Eval
    pred_log = xgb_best.predict(X_val_enc)
    pred = np.expm1(pred_log)
    rmse = np.sqrt(((pred - y_val) ** 2).mean())
    mae = np.mean(np.abs(pred - y_val))
    print(f"🏁 Holdout – RMSE: {rmse:,.2f} | MAE: {mae:,.2f}")

    # === Optional: Multi-Seed-Blending auf Holdout ===
    print("🎲 Starte Seed-Blending (5 Modelle) …")
    seeds = [11, 22, 33, 44, 55]
    blend_preds = []
    for sd in seeds:
        model = XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=sd,
        )
        try:
            model.fit(
                X_tr_enc, y_tr_log,
                eval_set=[(X_val_enc, y_val_log)],
                callbacks=[xgb.callback.EarlyStopping(rounds=80, save_best=True, maximize=False)]
            )
        except TypeError:
            try:
                model.fit(
                    X_tr_enc, y_tr_log,
                    eval_set=[(X_val_enc, y_val_log)],
                    early_stopping_rounds=80,
                    verbose=False
                )
            except TypeError:
                model.fit(X_tr_enc, y_tr_log)
        blend_preds.append(np.expm1(model.predict(X_val_enc)))

    blend_pred = np.mean(blend_preds, axis=0)
    blend_rmse = np.sqrt(((blend_pred - y_val) ** 2).mean())
    print(f"🧩 Blend RMSE: {blend_rmse:,.2f}")

    # Modell speichern
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "preprocessor": pre,
        "model": xgb_best,
        "target": TARGET,
        "use_log_target": True,
        "best_params": best_params
    }, MODEL_PATH)
    print(f"💾 Modell gespeichert: {MODEL_PATH}")

    # === Submission (Seed-Blend auf Test) ===
    test_path = DATA_PATH.parent / "test.csv"
    if test_path.exists():
        print("📝 Erzeuge Submission aus getuntem Modell …")
        x_test = pd.read_csv(test_path)
        x_test = build_features(x_test)
        X_test_enc = pre.transform(x_test)

        test_blends = []
        for sd in seeds:
            model = XGBRegressor(
                **best_params,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
                random_state=sd,
            )
            try:
                model.fit(
                    X_tr_enc, y_tr_log,
                    eval_set=[(X_val_enc, y_val_log)],
                    callbacks=[xgb.callback.EarlyStopping(rounds=80, save_best=True, maximize=False)]
                )
            except TypeError:
                try:
                    model.fit(
                        X_tr_enc, y_tr_log,
                        eval_set=[(X_val_enc, y_val_log)],
                        early_stopping_rounds=80,
                        verbose=False
                    )
                except TypeError:
                    model.fit(X_tr_enc, y_tr_log)
            test_blends.append(np.expm1(model.predict(X_test_enc)))
        preds = np.mean(test_blends, axis=0)

        submissions_dir = PROJECT_ROOT / "Submissions"
        submissions_dir.mkdir(exist_ok=True)
        submission_path = submissions_dir / "submission_xgb_tuned.csv"
        pd.DataFrame({"Id": x_test["Id"], "SalePrice": preds}).to_csv(submission_path, index=False)
        print(f"📦 Submission gespeichert: {submission_path}")
    else:
        print("⚠️ Testdatei nicht gefunden – keine Submission erstellt.")

    # === Summary Printout ===
    submission_path_str = str(submission_path) if 'submission_path' in locals() else "– keine –"
    print("\n📊 Zusammenfassung:")
    print("────────────────────────────")
    print(f"Holdout RMSE: {rmse:,.2f}")
    print(f"Holdout MAE : {mae:,.2f}")
    print(f"Blend RMSE  : {blend_rmse:,.2f}")
    print("────────────────────────────")
    print(f"Best Params : {best_params}")
    print(f"Modell      : {MODEL_PATH.name}")
    print(f"Submission  : {submission_path_str}")
    print("────────────────────────────")


if __name__ == "__main__":
    main()
