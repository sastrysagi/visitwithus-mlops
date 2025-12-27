# deployment/push_to_hf_space.py
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


def ensure_readme():
    """
    Spaces REQUIRE a README.md with YAML front-matter config.
    This creates/overwrites a correct one in repo root.
    """
    readme_path = Path("README.md")
    readme_text = """---
title: VisitWithUs - Wellness Package Purchase Predictor
emoji: 🧳
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
---

# VisitWithUs - Wellness Package Purchase Predictor

This Streamlit app predicts whether a customer will purchase the wellness package (**ProdTaken**).
"""
    readme_path.write_text(readme_text, encoding="utf-8")
    print("[INFO] README.md written with valid Spaces config.")


def main():
    hf_token = os.getenv("HF_TOKEN")
    hf_space_repo = os.getenv("HF_SPACE_REPO")  # ex: "sastrysagi/visitwithus-mlops"
    assert hf_token, "HF_TOKEN is missing in environment/secrets."
    assert hf_space_repo, "HF_SPACE_REPO is missing in environment/secrets. Example: username/space_name"

    api = HfApi(token=hf_token)

    # 1) Ensure Space exists (create if missing)
    try:
        api.repo_info(repo_id=hf_space_repo, repo_type="space")
        print(f"[INFO] HF Space exists: {hf_space_repo}")
    except RepositoryNotFoundError:
        print(f"[INFO] HF Space not found. Creating: {hf_space_repo}")
        create_repo(
            repo_id=hf_space_repo,
            repo_type="space",
            space_sdk="streamlit",
            private=False,
            token=hf_token,
        )
        print(f"[INFO] Created HF Space: {hf_space_repo}")
    except HfHubHTTPError as e:
        # sometimes HF returns 404/other for permissions; show clean error
        raise RuntimeError(f"HF Space repo_info failed: {e}") from e

    # 2) Ensure README front-matter exists
    ensure_readme()

    # 3) Validate model path (this MUST exist before pushing)
    # Recommended standard path in repo:
    #   artifacts/model/model.joblib
    model_path = Path("artifacts/model/model.joblib")
    if not model_path.exists():
        # fallback if you saved it differently
        alt1 = Path("artifacts/model.joblib")
        alt2 = Path("artifacts/model/model.pkl")
        if alt1.exists():
            model_path = alt1
        elif alt2.exists():
            model_path = alt2
        else:
            raise FileNotFoundError(
                "Model not found. Expected one of:\n"
                "- artifacts/model/model.joblib (recommended)\n"
                "- artifacts/model.joblib\n"
                "- artifacts/model/model.pkl\n"
                "Run model training first and ensure model is saved."
            )

    print(f"[INFO] Using model file: {model_path}")

    # 4) Upload only the required files for Space
    # NOTE: upload_folder expects repo exists already.
    allow_patterns = [
        "app.py",
        "requirements.txt",
        "README.md",
        str(model_path).replace("\\", "/"),
    ]

    print(f"[INFO] Uploading app + model to HF Space: {hf_space_repo}")
    api.upload_folder(
        folder_path=".",
        repo_id=hf_space_repo,
        repo_type="space",
        allow_patterns=allow_patterns,
        commit_message="Deploy VisitWithUs Streamlit app + model",
    )

    print("[SUCCESS] Deployed to Hugging Face Space:", hf_space_repo)


if __name__ == "__main__":
    main()

