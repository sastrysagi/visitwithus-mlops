# deployment/push_to_hf_space.py
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub.utils._http import HfHubHTTPError

SPACE_SDK = "streamlit"
SPACE_SDK_VERSION = "1.33.0"  # safe default

README_TEMPLATE = """---
title: VisitWithUs - Wellness Package Purchase Predictor
emoji: 🧳
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "{sdk_version}"
app_file: app.py
pinned: false
---

# VisitWithUs - Wellness Package Purchase Predictor

Streamlit app to predict whether a customer will purchase the wellness package (ProdTaken).
"""


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def ensure_space_exists(api: HfApi, space_repo: str, token: str):
    try:
        api.repo_info(repo_id=space_repo, repo_type="space")
        print(f"[INFO] Space exists: {space_repo}")
    except RepositoryNotFoundError:
        print(f"[INFO] Space not found. Creating: {space_repo}")
        create_repo(
            repo_id=space_repo,
            repo_type="space",
            private=False,
            space_sdk=SPACE_SDK,
            token=token,
        )
        print(f"[INFO] Space created: {space_repo}")
    except HfHubHTTPError as e:
        # If 404/other, still try create
        if "404" in str(e):
            print(f"[WARN] Space not found (404). Creating: {space_repo}")
            create_repo(
                repo_id=space_repo,
                repo_type="space",
                private=False,
                space_sdk=SPACE_SDK,
                token=token,
            )
            print(f"[INFO] Space created: {space_repo}")
        else:
            raise


def write_readme_if_missing(dst_root: Path):
    readme = dst_root / "README.md"
    if not readme.exists():
        readme.write_text(README_TEMPLATE.format(sdk_version=SPACE_SDK_VERSION), encoding="utf-8")
        print("[INFO] README.md created with Spaces config.")
    else:
        # Ensure it has the required front-matter; if not, overwrite safely
        txt = readme.read_text(encoding="utf-8", errors="ignore")
        if not txt.lstrip().startswith("---"):
            readme.write_text(README_TEMPLATE.format(sdk_version=SPACE_SDK_VERSION), encoding="utf-8")
            print("[WARN] README.md existed but had no config. Overwritten with Spaces config.")


def main():
    hf_token = _require_env("HF_TOKEN")
    space_repo = _require_env("HF_SPACE_REPO")  # e.g. "sastrysagi/visitwithus-mlops-ui"

    api = HfApi(token=hf_token)

    # 1) Ensure the Space exists
    ensure_space_exists(api, space_repo, hf_token)

    repo_root = Path(".").resolve()

    # 2) Validate required files locally
    app_py = repo_root / "app.py"
    if not app_py.exists():
        raise FileNotFoundError("app.py not found in repo root. Space needs app.py at root.")

    # Model artifact locations we support
    candidate_models = [
        repo_root / "artifacts" / "model.joblib",
        repo_root / "artifacts" / "model" / "model.joblib",
        repo_root / "artifacts" / "model" / "best_model.joblib",
    ]
    model_path = next((p for p in candidate_models if p.exists()), None)
    if not model_path:
        raise FileNotFoundError(
            "Model file not found. Expected one of:\n"
            " - artifacts/model.joblib\n"
            " - artifacts/model/model.joblib\n"
            " - artifacts/model/best_model.joblib\n"
            "Make sure your training job uploads the model artifact into repo before deploy."
        )
    print(f"[INFO] Found model artifact: {model_path}")

    # 3) Create a clean temp folder to upload to Space
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Copy app.py
        shutil.copy2(app_py, tmp / "app.py")

        # Copy requirements.txt (Space needs it)
        req = repo_root / "requirements.txt"
        if req.exists():
            shutil.copy2(req, tmp / "requirements.txt")
        else:
            # Fallback minimal requirements if not present
            (tmp / "requirements.txt").write_text(
                "streamlit\npandas\nnumpy\nscikit-learn==1.3.2\njoblib\n",
                encoding="utf-8",
            )
            print("[WARN] requirements.txt missing; wrote fallback with sklearn pin.")

        # Copy artifacts folder (keep exact expected path)
        # We will normalize to artifacts/model.joblib for app loading simplicity
        artifacts_dir = tmp / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, artifacts_dir / "model.joblib")
        print("[INFO] Copied model to Space payload: artifacts/model.joblib")

        # Optional: copy any extra assets (like label encoders etc.)
        extra_dirs = ["reports"]
        for d in extra_dirs:
            src = repo_root / d
            if src.exists() and src.is_dir():
                shutil.copytree(src, tmp / d, dirs_exist_ok=True)

        # 4) Ensure README config exists
        write_readme_if_missing(tmp)

        # 5) Push folder to Space
        print(f"[INFO] Uploading Space payload to: {space_repo}")
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=space_repo,
            repo_type="space",
            commit_message="Deploy VisitWithUs Streamlit app + model artifact",
        )
        print(f"[SUCCESS] Deployed to HF Space: {space_repo}")


if __name__ == "__main__":
    main()

