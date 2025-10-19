from pathlib import Path
import io
import numpy as np
import pandas as pd
import joblib

import streamlit as st
import matplotlib.pyplot as plt

# Optional (SHAP ist nice-to-have, aber nicht Pflicht)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ====== Pfade / Settings ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../HousePrices/HousePrices
DATA_DIR = PROJECT_ROOT / "data" / "house-prices-advanced-regression-techniques"
MODEL_DEFAULT = PROJECT_ROOT / "models" / "xgb_tuned_log.pkl"  # dein bestes Modell
SUBMISSIONS_DIR = PROJECT_ROOT / "Submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "SalePrice"

# Feature Engineering importieren
from feature_engineering import build_features


# ---------- Helpers ----------
def load_bundle(path: Path):
    bundle = joblib.load(path)
    if "preprocessor" in bundle and "model" in bundle:
        pre = bundle["preprocessor"]
        model = bundle["model"]
        use_log = bool(bundle.get("use_log_target", False))
        return pre, model, use_log
    elif "pipeline" in bundle:
        # Modelle aus train_all: komplette Pipeline
        return bundle["pipeline"], None, False
    else:
        raise ValueError("Unbekanntes Bundle-Format.")

def ensure_dataframe(X_enc, feat_names=None):
    """Sorge für konsistente Feature-Namen (für LGBM/XGB mit Namenstracking)."""
    try:
        import scipy.sparse as sp
        if sp.issparse(X_enc):
            X_enc = X_enc.toarray()
    except Exception:
        pass
    if feat_names is None:
        feat_names = [f"f{i}" for i in range(X_enc.shape[1])]
    return pd.DataFrame(X_enc, columns=feat_names)

def predict_with_bundle(pre, model, use_log, df_feat: pd.DataFrame):
    """Preprocess + predict; unterstützt sowohl (pre, model) als auch pipeline-only."""
    if hasattr(pre, "transform") and model is not None:
        X_enc = pre.transform(df_feat)
        try:
            feat_names = pre.get_feature_names_out()
        except Exception:
            feat_names = None
        X_enc_df = ensure_dataframe(X_enc, feat_names)
        preds = model.predict(X_enc_df)
    else:
        # pipeline-only
        preds = pre.predict(df_feat)

    if use_log:
        preds = np.expm1(preds)
    return preds

def metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return mae, rmse, r2

def download_link(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name=filename, mime="text/csv")


# ---------- UI ----------
st.set_page_config(page_title="House Prices – ML Dashboard", layout="wide")
st.title("🏠 House Prices – ML Dashboard (lokal)")

# Sidebar: Modell wählen
st.sidebar.header("Modell & Daten")
model_path = st.sidebar.text_input("Pfad zum Modell (.pkl)", str(MODEL_DEFAULT))
model_path = Path(model_path)

# Datenquelle wählen
data_choice = st.sidebar.selectbox(
    "Datenquelle",
    ["Kaggle train.csv (mit Labels)", "Kaggle test.csv (ohne Labels)", "Eigenes CSV hochladen"]
)

sample_n = st.sidebar.slider("Sample-Größe (nur Anzeige/Plots)", min_value=200, max_value=1460, value=800, step=50)

# ---------- Load Model ----------
try:
    pre, model, use_log = load_bundle(model_path)
    st.success(f"Modell geladen: {model_path.name}")
except Exception as e:
    st.error(f"Modell konnte nicht geladen werden: {e}")
    st.stop()

# ---------- Load Data ----------
df = None
gt_available = False
if data_choice.startswith("Kaggle train"):
    path = DATA_DIR / "train.csv"
    df = pd.read_csv(path)
    gt_available = True
elif data_choice.startswith("Kaggle test"):
    path = DATA_DIR / "test.csv"
    df = pd.read_csv(path)
else:
    upl = st.sidebar.file_uploader("Eigenes CSV wählen", type=["csv"])
    if upl is not None:
        df = pd.read_csv(upl)

if df is None:
    st.info("Bitte Datenquelle wählen oder CSV hochladen.")
    st.stop()

# ---------- Feature Engineering ----------
df_raw = df.copy()
df_feat = build_features(df.copy())

# ---------- Predict ----------
if TARGET in df_feat.columns:
    X = df_feat.drop(columns=[TARGET])
    y_true = df_feat[TARGET].astype(float).values
    gt_available = True
else:
    X = df_feat
    y_true = None
    gt_available = False

preds = predict_with_bundle(pre, model, use_log, X)
preds = np.clip(preds, 1.0, None)  # Sanity

# ---------- Zusammenführung & Metriken ----------
out = df_raw.copy()
out["Predicted_SalePrice"] = preds
if "Id" in out.columns and out["Id"].dtype != int:
    out["Id"] = out["Id"].astype(int)

if gt_available:
    mae, rmse, r2 = metrics(y_true, preds)
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:,.0f}")
    c2.metric("RMSE", f"{rmse:,.0f}")
    c3.metric("R²", f"{r2:.3f}")
else:
    med = float(np.median(preds))
    st.metric("Median Vorhersage", f"{med:,.0f}")

# ---------- Plots ----------
st.subheader("Verteilung der Vorhersagen")
st.bar_chart(pd.DataFrame({"SalePrice": preds}).sample(min(sample_n, len(preds))))

if gt_available:
    st.subheader("Actual vs. Predicted (Sample)")
    sz = min(sample_n, len(out))
    sample = out.head(sz).copy()
    fig, ax = plt.subplots()
    ax.scatter(sample.index, sample[TARGET], label="Actual", alpha=0.6)
    ax.scatter(sample.index, sample["Predicted_SalePrice"], label="Predicted", alpha=0.6)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("SalePrice")
    ax.legend()
    st.pyplot(fig)

    # Top-Über/Unter-Schätzer
    st.subheader("Größte Abweichungen")
    sample["AbsErr"] = np.abs(sample[TARGET] - sample["Predicted_SalePrice"])
    st.dataframe(sample.sort_values("AbsErr", ascending=False).head(20)[["Id", TARGET, "Predicted_SalePrice", "AbsErr"]])

# ---------- SHAP ----------
st.subheader("Feature-Erklärung (SHAP)")
if HAS_SHAP and model is not None:
    try:
        # encodieren + Namen holen
        X_enc = pre.transform(X)
        try:
            feat_names = pre.get_feature_names_out()
        except Exception:
            feat_names = None
        X_enc_df = ensure_dataframe(X_enc, feat_names)

        # subsample für Speed (mit Namen behalten!)
        n_show = min(500, len(X_enc_df))
        X_show = X_enc_df.iloc[:n_show]  # <<< WICHTIG: KEIN .values, sonst Namen weg

        # kleiner Background für Kernel-/TreeExplainer, verbessert Stabilität & Speed
        bg_size = min(1000, len(X_enc_df))
        X_bg = X_enc_df.sample(bg_size, random_state=42) if len(X_enc_df) > bg_size else X_enc_df

        explainer = shap.Explainer(model, X_bg)
        shap_values = explainer(X_show)

        st.write("Top-Features (bar):")
        fig1, ax1 = plt.subplots()
        shap.plots.bar(shap_values, show=False, max_display=20)
        st.pyplot(fig1)
        plt.close(fig1)

        st.write("Verteilung (beeswarm):")
        fig2, ax2 = plt.subplots()
        shap.plots.beeswarm(shap_values, show=False, max_display=20)
        st.pyplot(fig2)
        plt.close(fig2)

    except Exception as e:
        st.info(f"SHAP aktuell nicht verfügbar: {e}")
else:
    st.caption("SHAP nicht installiert oder Modell nicht kompatibel. Optional: `pip install shap`.")

# ---------- Tabelle & Export ----------
st.subheader("Tabelle (Sample)")
st.dataframe(out.head(min(sample_n, len(out))))

# Submission/Export-Buttons
c1, c2 = st.columns(2)
with c1:
    if "Id" in out.columns:
        sub = out[["Id", "Predicted_SalePrice"]].rename(columns={"Predicted_SalePrice": "SalePrice"})
        download_link(sub, "submission_dashboard.csv")
with c2:
    download_link(out, "predictions_with_features.csv")

st.caption("Hinweis: Für Kaggle-Upload brauchst du Spalten `Id, SalePrice` ohne weitere Felder.")