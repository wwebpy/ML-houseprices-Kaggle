from pathlib import Path
import pandas as pd
import numpy as np
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python src/compare_submissions.py Submissions/submission_prev.csv Submissions/submission_xgb_tuned.csv")
        return
    p1, p2 = Path(sys.argv[1]), Path(sys.argv[2])
    s1 = pd.read_csv(p1).rename(columns={"SalePrice":"SalePrice_prev"})
    s2 = pd.read_csv(p2).rename(columns={"SalePrice":"SalePrice_new"})
    df = s1.merge(s2, on="Id", how="inner")
    df["diff"] = df["SalePrice_new"] - df["SalePrice_prev"]
    df["abs_diff"] = df["diff"].abs()

    print(f"Gemeinsame IDs: {len(df)}")
    print(f"Ø Differenz: {df['diff'].mean():,.2f}")
    print(f"Ø |Differenz|: {df['abs_diff'].mean():,.2f}")
    print(f"Max |Differenz|: {df['abs_diff'].max():,.2f}")
    print(f"Korrelationskoeff.: {df['SalePrice_new'].corr(df['SalePrice_prev']):.4f}")

    out = Path("Submissions/diff_report.csv")
    df.to_csv(out, index=False)
    print(f"Detail-Report gespeichert: {out}")

    # Optional: einfacher Blend (kann oft minimal verbessern)
    blend = df[["Id"]].copy()
    w = 0.5  # Gewichtung anpassen, z.B. 0.6/0.4
    blend["SalePrice"] = w*df["SalePrice_new"] + (1-w)*df["SalePrice_prev"]
    blend_out = Path("Submissions/submission_blend.csv")
    blend.to_csv(blend_out, index=False)
    print(f"Blend gespeichert: {blend_out}")

if __name__ == "__main__":
    main()