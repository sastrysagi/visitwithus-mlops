from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from pathlib import Path
import os

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found. Add it to Colab Secrets or environment variables.")

HF_USERNAME = os.getenv("HF_USERNAME")
HF_SPACE_REPO = os.getenv("HF_SPACE_REPO")  # e.g. 'username/visitwithus-streamlit'
if not HF_SPACE_REPO:
    if not HF_USERNAME:
        raise RuntimeError("Set HF_SPACE_REPO or HF_USERNAME (and it will default repo name).")
    HF_SPACE_REPO = f"{HF_USERNAME}/visitwithus-streamlit"

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=HF_SPACE_REPO, repo_type="space")
    print("Space exists:", HF_SPACE_REPO)
except RepositoryNotFoundError:
    print("Creating space:", HF_SPACE_REPO)
    create_repo(
        repo_id=HF_SPACE_REPO,
        repo_type="space",
        space_sdk="streamlit",
        private=False,
        token=HF_TOKEN,
    )

project_dir = Path("tourism_project")

# upload app + requirements
for name in ["app.py", "requirements.txt"]:
    api.upload_file(
        path_or_fileobj=str(project_dir / name),
        path_in_repo=name,
        repo_id=HF_SPACE_REPO,
        repo_type="space",
    )

# upload artifacts so the app can load the model
api.upload_folder(
    folder_path=str(project_dir / "artifacts"),
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    path_in_repo="artifacts",
)

print("Deployed Streamlit Space to:", HF_SPACE_REPO)


