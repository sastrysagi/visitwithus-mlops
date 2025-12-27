# deployment/push_to_hf_space.py
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub.utils import HfHubHTTPError


def _required_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise EnvironmentError(f"{name} is missing. Set it as a GitHub Actions secret/env.")
    return v


def ensure_space_exists(api: HfApi, repo_id: str, token: str):
    # repo_id must be like "username/space_name"
    if "/" not in repo_id:
        raise ValueError(f"HF_SPACE_REPO must be like 'username/space_name'. Got: {repo_id}")

    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"[INFO] HF Space exists: {repo_id}")
        return
    except RepositoryNotFoundError:
        print(f"[INFO] HF Space not found. Creating: {repo_id}")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="space",
                token=token,
                private=False,
                space_sdk="streamlit",   # IMPORTANT
            )
            print(f"[INFO] Created Space: {repo_id}")
        except HfHubHTTPError as e:
            # If it already exists / race condition
            if "409" in str(e):
                print(f"[WARN] Space already exists (409). Continuing: {repo_id}")
            else:
                raise


def build_readme(space_title: str) -> str:
    # HF Spaces require this YAML header
    return f"""---
title: {space_title}
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
---

# {space_title}

Streamlit app deployed via GitHub Actions.
"""


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[INFO] Copied: {src} -> {dst}")
        return True
    return False


def main():
    hf_token = _required_env("HF_TOKEN")
    hf_space_repo = _required_env("HF_SPACE_REPO")

    # Optional (nice-to-have)
    space_title = os.getenv("HF_SPACE_TITLE", "VisitWithUs - Wellness Package Purchase Predictor")

    api = HfApi(token=hf_token)
    ensure_space_exists(api, repo_id=hf_space_repo, token=hf_token)

    # Prepare a clean deploy folder (what will be uploaded to Space)
    deploy_dir = Path("deployment") / "_hf_space_bundle"
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir(parents=True, exist_ok=True)

    # 1) app.py must exist at repo root (as your workflow uses)
    # If your app.py is at root, we copy it. If not, we try deployment/app.py.
    root_app = Path("app.py")
    alt_app = Path("deployment") / "app.py"
    if root_app.exists():
        shutil.copy2(root_app, deploy_dir / "app.py")
    elif alt_app.exists():
        shutil.copy2(alt_app, deploy_dir / "app.py")
    else:
        raise FileNotFoundError("app.py not found at repo root or deployment/app.py")

    # 2) requirements.txt
    req = Path("requirements.txt")
    if not req.exists():
        raise FileNotFoundError("requirements.txt not found at repo root.")
    shutil.copy2(req, deploy_dir / "requirements.txt")

    # 3) README.md with HF Space config header
    (deploy_dir / "README.md").write_text(build_readme(space_title), encoding="utf-8")

    # 4) Ensure model file exists in Space at artifacts/model.joblib
    # Your Streamlit error expects: /app/artifacts/model.joblib
    # We will build exactly that path in the Space repo.
    model_dst = deploy_dir / "artifacts" / "model.joblib"

    # Common places where training may save the model:
    candidates = [
        Path("artifacts") / "model.joblib",
        Path("artifacts") / "model" / "model.joblib",
        Path("artifacts") / "model" / "best_model.joblib",
        Path("model_building") / "artifacts" / "model.joblib",
        Path("model_building") / "artifacts" / "model" / "model.joblib",
    ]

    found = False
    for c in candidates:
        if copy_if_exists(c, model_dst):
            found = True
            break

    if not found:
        raise FileNotFoundError(
            "Could not find a trained model file to deploy. "
            "Expected one of:\n" + "\n".join([str(c) for c in candidates]) +
            "\n\nMake sure model-training job saved the model first."
        )

    # (Optional) If you have any extra artifacts needed by app.py (encoders/feature list), add here.
    # Example:
    # copy_if_exists(Path("artifacts") / "feature_schema.json", deploy_dir / "artifacts" / "feature_schema.json")

    print(f"[INFO] Uploading Space bundle folder: {deploy_dir}")

    api.upload_folder(
        folder_path=str(deploy_dir),
        repo_id=hf_space_repo,
        repo_type="space",
        commit_message="Deploy Streamlit app + model via CI",
        token=hf_token,
    )

    print(f"[SUCCESS] Deployed to HF Space: {hf_space_repo}")


if __name__ == "__main__":
    main()
