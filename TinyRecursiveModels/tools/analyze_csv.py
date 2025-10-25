import os
import argparse
import pandas as pd
import numpy as np


def analyze_csv(path: str, threshold: float, out_path: str | None):
    df = pd.read_csv(path)
    n_rows = len(df)
    print(f"Loaded: {path} rows={n_rows} cols={len(df.columns)}")

    # Basic per-column stats
    rows = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce") if not pd.api.types.is_numeric_dtype(df[col]) else df[col]
        missing = int(s.isna().sum())
        missing_ratio = float(missing / n_rows)
        nunique = int(df[col].nunique(dropna=True))
        is_numeric = pd.api.types.is_numeric_dtype(s)
        mean = float(s.mean()) if is_numeric else np.nan
        std = float(s.std()) if is_numeric else np.nan
        minv = float(s.min()) if is_numeric else np.nan
        maxv = float(s.max()) if is_numeric else np.nan
        constant = (nunique == 1)
        all_nan = (missing == n_rows)
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "nunique": nunique,
            "missing": missing,
            "missing_ratio": missing_ratio,
            "constant": constant,
            "all_nan": all_nan,
            "mean": mean,
            "std": std,
            "min": minv,
            "max": maxv,
        })

    rep = pd.DataFrame(rows).sort_values("missing_ratio", ascending=False)
    print("\nTop 30 columns by missing ratio:")
    print(rep[["column", "missing", "missing_ratio", "dtype", "nunique", "constant", "all_nan"]].head(30).to_string(index=False))

    high_missing = rep[rep["missing_ratio"] >= threshold]["column"].tolist()
    print(f"\nColumns with missing_ratio >= {threshold:.2f} (count={len(high_missing)}):")
    if high_missing:
        print(", ".join(high_missing))
    else:
        print("(none)")

    n_rows_any_missing = int(df.isna().any(axis=1).sum())
    print(f"\nRows with any missing values: {n_rows_any_missing} ({n_rows_any_missing/n_rows:.2%})")

    # Save full report if requested
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        rep.to_csv(out_path, index=False)
        print(f"\nSaved full report to: {out_path}")

    # Optional: correlation with forward_returns if present
    if "forward_returns" in df.columns:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        corr = df[num_cols].corr(numeric_only=True)["forward_returns"].sort_values(ascending=False)
        print("\nTop 15 absolute correlations with forward_returns:")
        print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(15).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze train.csv for missingness and basic stats")
    parser.add_argument("--path", type=str, default=os.path.join("TinyRecursiveModels", "train.csv"))
    parser.add_argument("--threshold", type=float, default=0.3, help="missing ratio threshold to flag columns")
    parser.add_argument("--out", type=str, default=os.path.join("TinyRecursiveModels", "na_report.csv"))
    args = parser.parse_args()
    analyze_csv(args.path, args.threshold, args.out)


