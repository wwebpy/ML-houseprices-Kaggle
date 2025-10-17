from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ==== Pfade ====
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "house-prices-advanced-regression-techniques" / "train.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "rf_regressor.pkl"

TARGET = "SalePrice"   # <- Zielvariable

def main():
    # === 1. Daten laden ===
    df = pd.read_csv(DATA_PATH)
    print(f"Daten geladen: {df.shape}")
    print(df.head(3))

    # === 2. Features / Target trennen ===
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # === 3. Spaltentypen bestimmen ===
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"Numerisch: {len(num_cols)} | Kategorisch: {len(cat_cols)}")

    # === 4. Preprocessing definieren ===
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    # === 5. Modell definieren ===
    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    pipe = Pipeline([
        ("pre", pre),
        ("model", model)
    ])

    # === 6. Train-Test-Split ===
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === 7. Training ===
    pipe.fit(X_train, y_train)

    # === 8. Evaluation ===
    y_pred = pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"MAE: {mae:,.2f}")
    print(f"R² : {r2:.3f}")

    # === 9. Modell speichern ===
    joblib.dump({
        "pipeline": pipe,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target": TARGET
    }, MODEL_PATH)
    print(f"✅ Modell gespeichert unter: {MODEL_PATH}")

if __name__ == "__main__":
    main()