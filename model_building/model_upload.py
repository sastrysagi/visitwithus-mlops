from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from pathlib import Path
import os

PROJECT_DIR = Path("tourism_project")
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found. Add it to Colab Secrets or environment variables.")

HF_USERNAME = os.getenv("HF_USERNAME")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")  # e.g. 'username/visitwithus-prod-taken-model'
if not HF_MODEL_REPO:
    if not HF_USERNAME:
        raise RuntimeError("Set HF_MODEL_REPO or HF_USERNAME (and it will default repo name).")
    HF_MODEL_REPO = f"{HF_USERNAME}/visitwithus-prod-taken-model"

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
    print("Model repo exists:", HF_MODEL_REPO)
except RepositoryNotFoundError:
    print("Creating model repo:", HF_MODEL_REPO)
    create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False, token=HF_TOKEN)

api.upload_folder(
    folder_path=str(ARTIFACTS_DIR),
    repo_id=HF_MODEL_REPO,
    repo_type="model",
)
print("Uploaded model artifacts to:", HF_MODEL_REPO)


