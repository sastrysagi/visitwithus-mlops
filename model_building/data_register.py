# model_building/data_register.py
import os
import re
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    HfHubHTTPError,
)

def _clean_repo_id(raw: str) -> str:
    """
    Accepts:
      - username/repo
      - https://huggingface.co/datasets/username/repo
      - datasets/username/repo
    Returns:
      - username/repo
    """
    if not raw or not isinstance(raw, str):
        raise ValueError("HF_DATASET_REPO is empty or not a string.")

    s = raw.strip()

    # If URL, strip domain + optional /datasets/
    s = re.sub(r"^https?://huggingface\.co/", "", s)
    s = re.sub(r"^datasets/", "", s)
    s = re.sub(r"^dataset/", "", s)
    s = re.sub(r"^spaces/", "", s)
    s = re.sub(r"^space/", "", s)

    # If they accidentally included 'datasets/username/repo'
    s = re.sub(r"^datasets/", "", s)

    # Final sanity
    if "/" not in s:
        raise ValueError(
            f"Invalid repo_id '{raw}'. Expected format 'username/repo'."
        )
    return s

def main():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing. Add it to GitHub Secrets / env.")

    hf_username = os.getenv("HF_USERNAME", "").strip() or None
    raw_repo = os.getenv("HF_DATASET_REPO", "").strip()

    if not raw_repo:
        if not hf_username:
            raise RuntimeError(
                "HF_DATASET_REPO is missing and HF_USERNAME not set. "
                "Set secrets HF_DATASET_REPO='username/repo' (recommended)."
            )
        raw_repo = f"{hf_username}/visitwithus-tourism-dataset"

    repo_id = _clean_repo_id(raw_repo)

    api = HfApi(token=hf_token)

    # Ensure repo exists (do NOT fail on already exists)
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"[INFO] HF dataset repo exists: {repo_id}")
    except RepositoryNotFoundError:
        print(f"[INFO] Dataset repo not found. Creating: {repo_id}")
        try:
            # exist_ok=True prevents 409 from failing
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                token=hf_token,
                exist_ok=True,
            )
            print(f"[INFO] Created dataset repo: {repo_id}")
        except HfHubHTTPError as e:
            # 409 conflict can still happen depending on hub state; treat as OK
            if "409" in str(e):
                print(f"[WARN] Repo already exists (409). Continuing: {repo_id}")
            else:
                raise

    # Upload local folder 'data' (repo root) to HF dataset repo
    local_data_dir = Path("data")
    if not local_data_dir.exists():
        raise RuntimeError(
            "Local folder 'data/' not found in repo. "
            "Your repo listing shows 'data/' exists, so this should not happen."
        )

    print(f"[INFO] Uploading folder '{local_data_dir}' to HF dataset repo: {repo_id}")
    try:
        api.upload_folder(
            folder_path=str(local_data_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="",
            commit_message="Upload VisitWithUs dataset files",
        )
    except RepositoryNotFoundError as e:
        # This is the key failure you hit (404 preupload)
        print("[ERROR] Hugging Face says the dataset repo is not found or not accessible.")
        print("        Check that HF_DATASET_REPO is exactly 'username/repo' (NOT a URL).")
        print("        Also ensure HF_TOKEN has permission to that user/org.")
        raise e

    print(f"[SUCCESS] Uploaded data/ to HF dataset repo: {repo_id}")

if __name__ == "__main__":
    main()
