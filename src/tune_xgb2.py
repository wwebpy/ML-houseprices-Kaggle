from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
import xgboost as xgb
from feature_engineering import build_features


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

    # CV-Statistiken des besten Kandidaten erfassen
    best_idx = search.best_index_
    cv_mean = -float(search.cv_results_["mean_test_score"][best_idx])
    cv_std = float(search.cv_results_["std_test_score"][best_idx])

    # === Retrain mit Early Stopping (Callbacks, kein eval_metric in fit/ctor) ===
    print("🚀 Retrain best params mit Early Stopping …")
    xgb_best = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    # Version-adaptive early stopping: try callbacks (xgboost>=2.0), fallback zu early_stopping_rounds (xgboost<2.0)
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
            # Fallback: ohne Early Stopping
            xgb_best.fit(X_tr_enc, y_tr_log)

    # Holdout-Eval
    pred_log = xgb_best.predict(X_val_enc)
    pred = np.expm1(pred_log)
    rmse = np.sqrt(((pred - y_val) ** 2).mean())
    mae = np.mean(np.abs(pred - y_val))
    print(f"🏁 Holdout – RMSE: {rmse:,.2f} | MAE: {mae:,.2f}")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_val, pred)
    print(f"➡️ R2 Score: {r2:.4f}")

    # === SHAP Feature Importance (TreeExplainer) ===
    try:
        import shap
        print("📊 Berechne SHAP Feature Importances …")

        # Ensure dense matrix for plotting
        try:
            import scipy.sparse as sp
            X_val_dense = X_val_enc.toarray() if sp.issparse(X_val_enc) else X_val_enc
        except Exception:
            X_val_dense = X_val_enc

        # Try to get feature names from the preprocessor
        try:
            feature_names = pre.get_feature_names_out()
        except Exception:
            # Fallback: generic names
            feature_names = np.array([f"f{i}" for i in range(X_val_dense.shape[1])])

        explainer = shap.TreeExplainer(xgb_best)
        shap_values = explainer(X_val_dense)
        try:
            shap_values.feature_names = feature_names
        except Exception:
            pass

        # Bar plot (global importance)
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        (PROJECT_ROOT/"models").mkdir(parents=True, exist_ok=True)
        plt.savefig(PROJECT_ROOT/"models"/"shap_xgb_bar.png", dpi=200)
        plt.close()

        # Beeswarm (distributional effects)
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT/"models"/"shap_xgb_beeswarm.png", dpi=200)
        plt.close()
        print("✅ SHAP Plots gespeichert: models/shap_xgb_bar.png, models/shap_xgb_beeswarm.png")
    except Exception as e:
        print(f"⚠️ SHAP Analyse übersprungen: {e}")

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

        # --- Sanity Guard: ensure predictions are in dollar-space and valid ---
        try:
            import numpy as _np
            med = _np.nanmedian(preds)
            if med < 1000:  # looks like log-space
                print("⚠️ Detected log-scale predictions for test — converting with expm1.")
                preds = _np.expm1(preds)

            # Ensure all predictions are finite and positive
            if not _np.isfinite(preds).all():
                print("⚠️ Non-finite predictions detected — fixing with nanmedian.")
                finite = preds[_np.isfinite(preds)]
                fallback = _np.nanmedian(finite) if finite.size else 180000.0
                preds = _np.where(_np.isfinite(preds), preds, fallback)
            preds = _np.clip(preds, 1.0, None)
        except Exception as _e:
            print(f"⚠️ Sanity guard skipped: {_e}")

        # Ensure Id is integer
        if x_test["Id"].dtype != int:
            x_test["Id"] = x_test["Id"].astype(int)

        submissions_dir = PROJECT_ROOT / "Submissions"
        submissions_dir.mkdir(exist_ok=True)
        submission_path = submissions_dir / "submission_xgb_tuned.csv"
        pd.DataFrame({"Id": x_test["Id"], "SalePrice": preds}).to_csv(submission_path, index=False)
        print(f"📦 Submission gespeichert: {submission_path}")
        # Post-save verification
        try:
            _chk = pd.read_csv(submission_path)
            print("🔎 Submission check — rows:", len(_chk), " median:", float(_chk["SalePrice"].median()))
        except Exception as _e:
            print(f"⚠️ Could not re-read submission for verification: {_e}")
    else:
        print("⚠️ Testdatei nicht gefunden – keine Submission erstellt.")

    # === Persistente Ergebnis-Tabelle (lokal) ===
    try:
        results_row = {
            "Model": "XGBoost",
            "Variant": "log",
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "CV_RMSE_mean": float(cv_mean),
            "CV_RMSE_std": float(cv_std),
            "Path": str(MODEL_PATH),
            "Submission": submission_path_str if 'submission_path_str' in locals() else "– keine –",
        }
        results_df = pd.DataFrame([results_row])

        results_file = PROJECT_ROOT / "results" / "results_tuned_xgb.csv"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        if results_file.exists():
            old = pd.read_csv(results_file)
            all_df = pd.concat([old, results_df], ignore_index=True)
            # Optional: Duplikate (gleiches Model+Variant+Path) entfernen
            all_df = all_df.drop_duplicates(subset=["Model", "Variant", "Path"], keep="last")
        else:
            all_df = results_df
        # Nach RMSE sortieren
        all_df = all_df.sort_values(by=["RMSE", "CV_RMSE_mean"]).reset_index(drop=True)
        all_df.to_csv(results_file, index=False)

        # Schöne Konsolenübersicht ähnlich train_all.py
        print("\nGesamtergebnis (lokal, sortiert nach RMSE):")
        with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
            cols = ["Model","Variant","MAE","RMSE","R2","CV_RMSE_mean","CV_RMSE_std","Path"]
            print(all_df[cols])
        print(f"\n💾 Lokale Ergebnisdatei aktualisiert: {results_file}")
    except Exception as e:
        print(f"⚠️ Konnte Ergebnisdatei nicht schreiben: {e}")

    # === Summary Printout ===
    submission_path_str = str(submission_path) if 'submission_path' in locals() else "– keine –"
    print("\n📊 Zusammenfassung:")
    print("────────────────────────────")
    print(f"Holdout RMSE: {rmse:,.2f}")
    print(f"Holdout MAE : {mae:,.2f}")
    print(f"Holdout R2  : {r2:.4f}")
    print(f"Blend RMSE  : {blend_rmse:,.2f}")
    print("────────────────────────────")
    print(f"Best Params : {best_params}")
    print(f"Modell      : {MODEL_PATH.name}")
    print(f"Submission  : {submission_path_str}")
    print("────────────────────────────")


if __name__ == "__main__":
    main()