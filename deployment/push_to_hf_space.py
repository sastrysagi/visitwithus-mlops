# deployment/push_to_hf_space.py
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub.utils._http import HfHubHTTPError


SPACE_README_TEMPLATE = """---
title: {title}
emoji: {emoji}
colorFrom: {colorFrom}
colorTo: {colorTo}
sdk: streamlit
sdk_version: "{sdk_version}"
app_file: app.py
pinned: false
---

# {title}

Streamlit app for **VisitWithUs Tourism Prediction**.

This Space is deployed automatically from GitHub Actions.
"""


def ensure_space_exists(api: HfApi, repo_id: str, token: str):
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"[INFO] Space repo exists: {repo_id}")
    except RepositoryNotFoundError:
        print(f"[INFO] Space repo not found. Creating: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            private=False,
            token=token,
            space_sdk="streamlit",  # IMPORTANT
        )
        print(f"[INFO] Space created: {repo_id}")
    except HfHubHTTPError as e:
        # Some org permissions / auth issues show up here
        raise RuntimeError(
            f"Failed checking/creating Space repo '{repo_id}'. "
            f"Verify HF_SPACE_REPO and token permissions. Error: {e}"
        )


def write_readme(space_dir: Path):
    readme_path = space_dir / "README.md"
    content = SPACE_README_TEMPLATE.format(
        title=os.getenv("HF_SPACE_TITLE", "VisitWithUs Tourism Predictor"),
        emoji=os.getenv("HF_SPACE_EMOJI", "🧳"),
        colorFrom=os.getenv("HF_SPACE_COLORFROM", "blue"),
        colorTo=os.getenv("HF_SPACE_COLORTO", "green"),
        sdk_version=os.getenv("HF_SPACE_SDK_VERSION", "1.36.0"),
    )
    readme_path.write_text(content, encoding="utf-8")
    print("[INFO] README.md written with Spaces config front-matter.")


def main():
    hf_token = os.getenv("HF_TOKEN")
    hf_space_repo = os.getenv("HF_SPACE_REPO")  # ex: "sastrysagi/visitwithus-mlops-space"

    if not hf_token:
        raise EnvironmentError("HF_TOKEN is missing. Add it to GitHub Secrets.")
    if not hf_space_repo:
        raise EnvironmentError("HF_SPACE_REPO is missing. Add it to GitHub Secrets (e.g., username/space-name).")

    api = HfApi(token=hf_token)

    # Ensure the Space repo exists (fixes 404)
    ensure_space_exists(api, hf_space_repo, hf_token)

    # Stage files to a temp folder to upload cleanly
    space_dir = Path("deployment") / "_hf_space_build"
    space_dir.mkdir(parents=True, exist_ok=True)

    # Required Space files
    # - app.py must exist at repo root for Space (as per README frontmatter)
    src_app = Path("app.py")
    if not src_app.exists():
        raise FileNotFoundError("app.py not found at repo root. Space needs app.py at root.")
    (space_dir / "app.py").write_text(src_app.read_text(encoding="utf-8"), encoding="utf-8")

    # requirements.txt (Space installs dependencies from this)
    src_req = Path("requirements.txt")
    if not src_req.exists():
        raise FileNotFoundError("requirements.txt not found at repo root.")
    (space_dir / "requirements.txt").write_text(src_req.read_text(encoding="utf-8"), encoding="utf-8")

    # README with HF Spaces config (fixes “Missing configuration in README”)
    write_readme(space_dir)

    # Include model artifacts (produced by training job)
    # Adjust if your model is saved differently.
    artifacts_model_dir = Path("artifacts") / "model"
    if artifacts_model_dir.exists():
        # Copy entire folder
        target_model_dir = space_dir / "artifacts" / "model"
        target_model_dir.mkdir(parents=True, exist_ok=True)
        for p in artifacts_model_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(artifacts_model_dir)
                out = target_model_dir / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(p.read_bytes())
        print("[INFO] Included model artifacts in Space upload.")
    else:
        print("[WARN] artifacts/model not found. Space will deploy but may not load model.")

    # Upload to HF Space
    print(f"[INFO] Uploading Space folder to: {hf_space_repo}")
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=hf_space_repo,
        repo_type="space",
        commit_message="CI: deploy VisitWithUs Streamlit Space",
    )

    print("[SUCCESS] Space deployed to Hugging Face:", hf_space_repo)


if __name__ == "__main__":
    main()

