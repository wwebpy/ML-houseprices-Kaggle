from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

from lightgbm import LGBMRegressor
from feature_engineering import build_features


# ===== Pfade & Settings =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "house-prices-advanced-regression-techniques" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "house-prices-advanced-regression-techniques" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_tuned.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
SUBMISSIONS_DIR = PROJECT_ROOT / "Submissions"

TARGET = "SalePrice"
RANDOM_STATE = 42


def main():
    # ===== Load =====
    if not DATA_PATH.exists():
        print(f"❌ Trainingsdatei fehlt: {DATA_PATH}")
        return

    print("📥 Lade Trainingsdaten …")
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        print(f"❌ Zielspalte '{TARGET}' nicht gefunden.")
        return

    df[TARGET] = df[TARGET].astype(float)

    # ===== Feature Engineering =====
    df = build_features(df)

    # (optionales) Topcap des Targets für Stabilität
    q_hi = df[TARGET].quantile(0.995)
    df.loc[df[TARGET] > q_hi, TARGET] = q_hi

    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET])

    # ===== Preprocessor =====
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    # ===== Split =====
    x_tr, x_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
    y_tr_log, y_val_log = np.log1p(y_tr), np.log1p(y_val)

    print("🧱 Fitte Preprocessing …")
    pre.fit(x_tr)
    X_tr_enc = pre.transform(x_tr)
    X_val_enc = pre.transform(x_val)

    # === Ensure consistent feature names to avoid sklearn/lightgbm warnings ===
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(X_tr_enc.shape[1])])

    def _to_df(m):
        try:
            import scipy.sparse as sp
            if sp.issparse(m):
                m = m.toarray()
        except Exception:
            pass
        return pd.DataFrame(m, columns=feat_names)

    X_tr_enc_df = _to_df(X_tr_enc)
    X_val_enc_df = _to_df(X_val_enc)

    # ===== Randomized Search (log target) =====
    param_dist = {
        "n_estimators": [600, 1000, 1500, 2000],  # kleinere Werte für schnellere CV
        "learning_rate": [0.03, 0.04, 0.05, 0.06],
        "num_leaves": [63, 95, 127, 191],
        "max_depth": [-1, 6, 8, 10],
        "min_child_samples": [5, 8, 10, 15, 20, 30],
        "min_split_gain": [0.0, 0.005, 0.01, 0.05],
        "reg_alpha": [0.0, 0.001, 0.01, 0.05, 0.1],
        "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "max_bin": [255, 511],
    }

    print("🔍 Starte RandomizedSearchCV (5-fold CV, log target) …")
    base_lgbm = LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_lgbm,
        param_distributions=param_dist,
        n_iter=40,
        scoring="neg_root_mean_squared_error",  # auf log-Target (numerisch kleiner = besser)
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    # Achtung: wir suchen auf log-Target mit encodierten Features
    search.fit(X_tr_enc_df, y_tr_log)
    best_params = search.best_params_

    print(f"✅ Beste CV-Params: {best_params}")
    print(f"✅ Beste CV-RMSE (log-space): {-search.best_score_:,.4f}")

    # ===== Real-Scale CV (Dollar-RMSE) =====
    # Stratify by price bins (real scale) for more stable CV
    try:
        bins = pd.qcut(y_tr, q=5, labels=False, duplicates='drop')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splitter = skf.split(X_tr_enc_df, bins)
    except Exception:
        splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X_tr_enc_df)
    cv_means_real = []
    for tr_idx, va_idx in splitter:
        Xtr, Xva = X_tr_enc_df.iloc[tr_idx], X_tr_enc_df.iloc[va_idx]
        ytr, yva = y_tr_log.iloc[tr_idx], y_tr_log.iloc[va_idx]

        mdl = LGBMRegressor(
            **best_params,
            objective="regression",
            boosting_type="gbdt",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        mdl.fit(Xtr, ytr)
        preds_log = mdl.predict(Xva)
        preds = np.expm1(preds_log)
        y_true = np.expm1(yva)
        rmse_fold = np.sqrt(((preds - y_true) ** 2).mean())
        cv_means_real.append(float(rmse_fold))

    cv_mean_real = float(np.mean(cv_means_real))
    cv_std_real = float(np.std(cv_means_real))
    print(f"💡 Real-scale CV RMSE (Dollar): {cv_mean_real:,.2f} ± {cv_std_real:,.2f}")

    # ===== Retrain best + Early Stopping =====
    print("🚀 Retrain best params mit Early Stopping …")
    lgbm_best = LGBMRegressor(
        **best_params,
        objective="regression",
        boosting_type="gbdt",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    # LGBM sklearn-API: early_stopping_rounds in fit + eval_set
    lgbm_best.fit(
        X_tr_enc_df, y_tr_log,
        eval_set=[(X_val_enc_df, y_val_log)],
        eval_metric="rmse",
        early_stopping_rounds=150,
        verbose=False
    )
    print("Best iteration:", getattr(lgbm_best, "best_iteration_", None))

    # ===== Holdout Metrics (Dollar) =====
    pred_log = lgbm_best.predict(X_val_enc_df)
    pred = np.expm1(pred_log)
    rmse = np.sqrt(((pred - y_val) ** 2).mean())
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)
    print(f"🏁 Holdout – RMSE: {rmse:,.2f} | MAE: {mae:,.2f}")
    print(f"➡️ R2 Score: {r2:.4f}")

    # ===== SHAP =====
    try:
        import shap
        print("📊 Berechne SHAP Feature Importances …")

        X_val_dense = X_val_enc_df.values

        try:
            feature_names = pre.get_feature_names_out()
        except Exception:
            feature_names = np.array([f"f{i}" for i in range(X_val_dense.shape[1])])

        explainer = shap.TreeExplainer(lgbm_best)
        shap_values = explainer(X_val_dense)
        try:
            shap_values.feature_names = feature_names
        except Exception:
            pass

        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        (PROJECT_ROOT/"models").mkdir(parents=True, exist_ok=True)
        plt.savefig(PROJECT_ROOT/"models"/"shap_lgbm_bar.png", dpi=200)
        plt.close()

        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT/"models"/"shap_lgbm_beeswarm.png", dpi=200)
        plt.close()
        print("✅ SHAP Plots gespeichert: models/shap_lgbm_bar.png, models/shap_lgbm_beeswarm.png")
    except Exception as e:
        print(f"⚠️ SHAP Analyse übersprungen: {e}")

    # ===== Seed-Blending (Holdout) =====
    print("🎲 Starte Seed-Blending (5 Modelle) …")
    seeds = [11, 22, 33, 44, 55]
    blend_preds = []
    for sd in seeds:
        model = LGBMRegressor(
            **best_params,
            objective="regression",
            boosting_type="gbdt",
            n_jobs=-1,
            random_state=sd
        )
        model.fit(
            X_tr_enc_df, y_tr_log,
            eval_set=[(X_val_enc_df, y_val_log)],
            eval_metric="rmse",
            early_stopping_rounds=120,
            verbose=False
        )
        blend_preds.append(np.expm1(model.predict(X_val_enc_df)))

    blend_pred = np.mean(blend_preds, axis=0)
    blend_rmse = np.sqrt(((blend_pred - y_val) ** 2).mean())
    print(f"🧩 Blend RMSE: {blend_rmse:,.2f}")

    # ===== Save Model =====
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "preprocessor": pre,
        "model": lgbm_best,
        "target": TARGET,
        "use_log_target": True,
        "best_params": best_params
    }, MODEL_PATH)
    print(f"💾 Modell gespeichert: {MODEL_PATH}")

    # ===== Submission =====
    submission_path = None
    if TEST_PATH.exists():
        print("📝 Erzeuge Submission (Seed-Blend auf Test) …")
        x_test = pd.read_csv(TEST_PATH)
        x_test = build_features(x_test)
        X_test_enc = pre.transform(x_test)
        X_test_enc_df = _to_df(X_test_enc)

        test_blends = []
        for sd in seeds:
            model = LGBMRegressor(
                **best_params,
                objective="regression",
                boosting_type="gbdt",
                n_jobs=-1,
                random_state=sd
            )
            model.fit(
                X_tr_enc_df, y_tr_log,
                eval_set=[(X_val_enc_df, y_val_log)],
                eval_metric="rmse",
                early_stopping_rounds=120,
                verbose=False
            )
            test_blends.append(np.expm1(model.predict(X_test_enc_df)))
        preds = np.mean(test_blends, axis=0)

        # Sanity Guard
        try:
            med = float(np.nanmedian(preds))
            if med < 1000:
                print("⚠️ Detected log-scale predictions for test — converting with expm1.")
                preds = np.expm1(preds)
            if not np.isfinite(preds).all():
                print("⚠️ Non-finite predictions detected — fixing with nanmedian.")
                finite = preds[np.isfinite(preds)]
                fallback = float(np.nanmedian(finite)) if finite.size else 180000.0
                preds = np.where(np.isfinite(preds), preds, fallback)
            preds = np.clip(preds, 1.0, None)
        except Exception as _e:
            print(f"⚠️ Sanity guard skipped: {_e}")

        if x_test["Id"].dtype != int:
            x_test["Id"] = x_test["Id"].astype(int)

        SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
        submission_path = SUBMISSIONS_DIR / "submission_lgbm_tuned.csv"
        pd.DataFrame({"Id": x_test["Id"], "SalePrice": preds}).to_csv(submission_path, index=False)
        print(f"📦 Submission gespeichert: {submission_path}")

        # post-save check
        try:
            _chk = pd.read_csv(submission_path)
            print("🔎 Submission check — rows:", len(_chk), " median:", float(_chk["SalePrice"].median()))
        except Exception as _e:
            print(f"⚠️ Could not re-read submission for verification: {_e}")
    else:
        print(f"⚠️ Testdatei nicht gefunden – {TEST_PATH}")

    # ===== Persist results (results_tuned_lgbm.csv) =====
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = RESULTS_DIR / "results_tuned_lgbm.csv"
        row = {
            "Model": "LightGBM",
            "Variant": "log",
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "CV_RMSE_mean": float(cv_mean_real),
            "CV_RMSE_std": float(cv_std_real),
            "Path": str(MODEL_PATH),
            "Submission": str(submission_path) if submission_path else "– keine –",
        }
        if out_file.exists():
            old = pd.read_csv(out_file)
            df_all = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
            df_all = df_all.drop_duplicates(subset=["Path"], keep="last")
        else:
            df_all = pd.DataFrame([row])
        df_all = df_all.sort_values(by=["RMSE", "CV_RMSE_mean"]).reset_index(drop=True)
        df_all.to_csv(out_file, index=False)
        print(f"💾 Ergebnisse gespeichert: {out_file}")
    except Exception as e:
        print(f"⚠️ Konnte Ergebnisdatei nicht schreiben: {e}")

    # ===== Summary =====
    submission_path_str = str(submission_path) if submission_path else "– keine –"
    print("\n📊 Zusammenfassung:")
    print("────────────────────────────")
    print(f"Holdout RMSE: {rmse:,.2f}")
    print(f"Holdout MAE : {mae:,.2f}")
    print(f"Holdout R2  : {r2:.4f}")
    print(f"Blend RMSE  : {blend_rmse:,.2f}")
    print("────────────────────────────")
    print(f"Real-CV RMSE: {cv_mean_real:,.2f} ± {cv_std_real:,.2f}")
    print(f"Best Params : {best_params}")
    print(f"Modell      : {MODEL_PATH.name}")
    print(f"Submission  : {submission_path_str}")
    print("────────────────────────────")


if __name__ == "__main__":
    main()