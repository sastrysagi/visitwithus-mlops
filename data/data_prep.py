# data/data_prep.py
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


RAW_LOCAL_PATHS = [
    Path("data/tourism.csv"),
    Path("tourism.csv"),
]

ARTIFACT_DIR = Path("artifacts/prepared")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREPARED = ARTIFACT_DIR / "prepared.csv"
OUT_CLEAN = ARTIFACT_DIR / "tourism_clean.csv"
OUT_REPORT = ARTIFACT_DIR / "data_validation_report.json"


def _normalize_gender_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = " ".join(s.split())  # collapse extra spaces e.g., "fe  male" -> "fe male"

    # Common normalizations / anomalies
    mapping = {
        "fe male": "female",
        "f e male": "female",
        "femail": "female",
        "fem ale": "female",
        "femal": "female",
        "woman": "female",
        "girl": "female",
        "m ale": "male",
        "ma le": "male",
        "mal e": "male",
        "man": "male",
        "boy": "male",
    }
    if s in mapping:
        return mapping[s]

    # If it contains the keyword female/male anywhere
    if "female" in s:
        return "female"
    if "male" in s:
        # NOTE: check male after female, because "female" contains "male" as substring
        return "male"

    # Unknown / other values remain as-is (you can also set to np.nan if required)
    return s


def load_raw_dataframe():
    # 1) Prefer local file in repo
    for p in RAW_LOCAL_PATHS:
        if p.exists():
            df = pd.read_csv(p)
            return df, f"local:{p.as_posix()}"

    # 2) Otherwise try HF dataset repo
    hf_repo = os.getenv("HF_DATASET_REPO")
    hf_token = os.getenv("HF_TOKEN")
    if hf_repo and hf_token and HF_AVAILABLE:
        api = HfApi(token=hf_token)

        # Try common filenames in dataset repo
        candidates = ["tourism.csv", "data/tourism.csv", "raw/tourism.csv"]
        for fname in candidates:
            try:
                url = api.hf_hub_url(repo_id=hf_repo, repo_type="dataset", filename=fname)
                df = pd.read_csv(url)
                return df, f"hf:{hf_repo}/{fname}"
            except Exception:
                continue

    raise FileNotFoundError(
        "Could not locate raw tourism.csv. "
        "Place it at data/tourism.csv OR set HF_TOKEN + HF_DATASET_REPO and ensure tourism.csv exists in the dataset repo."
    )


def validate_and_clean(df: pd.DataFrame):
    report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "shape_before": [int(df.shape[0]), int(df.shape[1])],
        "columns": {},
        "negative_values": {},
        "gender_cleanup": {},
        "missing_values": {},
        "dtype_summary": {},
    }

    # Dtype summary
    report["dtype_summary"] = {c: str(df[c].dtype) for c in df.columns}

    # Missing values
    report["missing_values"] = {c: int(df[c].isna().sum()) for c in df.columns}

    # Column-level profiling
    for c in df.columns:
        col = df[c]
        report["columns"][c] = {
            "dtype": str(col.dtype),
            "n_unique": int(col.nunique(dropna=True)),
            "sample_unique_values": (
                col.dropna().astype(str).unique()[:20].tolist()
                if col.dtype == "object" or str(col.dtype).startswith("string")
                else None
            ),
        }

    # Fix gender anomaly if a gender-like column exists
    gender_cols = [c for c in df.columns if c.strip().lower() in ["gender", "sex"]]
    if gender_cols:
        gcol = gender_cols[0]
        before_counts = df[gcol].astype(str).str.strip().str.lower().value_counts(dropna=False).to_dict()

        df[gcol] = df[gcol].apply(_normalize_gender_value)

        after_counts = df[gcol].astype(str).str.strip().str.lower().value_counts(dropna=False).to_dict()
        report["gender_cleanup"] = {
            "column": gcol,
            "value_counts_before": {k: int(v) for k, v in before_counts.items()},
            "value_counts_after": {k: int(v) for k, v in after_counts.items()},
        }
    else:
        report["gender_cleanup"] = {"column": None, "note": "No gender/sex column found."}

    # Negative value checks for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        neg_count = int((df[c] < 0).sum(skipna=True))
        if neg_count > 0:
            report["negative_values"][c] = {
                "negative_count": neg_count,
                "min_value": float(np.nanmin(df[c].values)),
            }

    # OPTIONAL: If you want to auto-fix negative values for columns that should never be negative:
    # Example: clip to 0
    # for c in report["negative_values"].keys():
    #     df[c] = df[c].clip(lower=0)

    report["shape_after"] = [int(df.shape[0]), int(df.shape[1])]
    return df, report


def upload_prepared_to_hf(clean_csv_path: Path, report_json_path: Path):
    hf_repo = os.getenv("HF_DATASET_REPO")
    hf_token = os.getenv("HF_TOKEN")

    if not (hf_repo and hf_token and HF_AVAILABLE):
        print("[INFO] HF upload skipped (HF_DATASET_REPO/HF_TOKEN missing or huggingface_hub not available).")
        return

    api = HfApi(token=hf_token)

    # Upload the cleaned outputs (THIS is what you wanted to see in HF dataset repo)
    print(f"[INFO] Uploading cleaned artifacts to HF dataset repo: {hf_repo}")
    api.upload_file(
        path_or_fileobj=str(clean_csv_path),
        path_in_repo="tourism_clean.csv",
        repo_id=hf_repo,
        repo_type="dataset",
        commit_message="Upload cleaned tourism dataset",
    )
    api.upload_file(
        path_or_fileobj=str(report_json_path),
        path_in_repo="data_validation_report.json",
        repo_id=hf_repo,
        repo_type="dataset",
        commit_message="Upload data validation report",
    )
    print("[INFO] HF upload done: tourism_clean.csv, data_validation_report.json")


def main():
    df, source = load_raw_dataframe()
    print(f"[INFO] Loaded raw data from {source} with shape={df.shape}")

    df_clean, report = validate_and_clean(df)

    # Save outputs
    df_clean.to_csv(OUT_CLEAN, index=False)
    df_clean.to_csv(OUT_PREPARED, index=False)  # training expects this name/path

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] Saved: {OUT_CLEAN}")
    print(f"[INFO] Saved: {OUT_PREPARED}")
    print(f"[INFO] Saved: {OUT_REPORT}")

    # Upload to HF dataset repo (clean outputs only)
    upload_prepared_to_hf(OUT_CLEAN, OUT_REPORT)


if __name__ == "__main__":
    main()

