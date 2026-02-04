from pathlib import Path

from huggingface_hub import HfFolder, upload_folder

from hf_utils import authenticate_hf

HF_TOKEN = HfFolder.get_token()
REPO_ID = "Malassila/eb-mia"
REPO_TYPE = "model"
LOCAL_DIR = Path("mia/results")
REMOTE_DIR = "results"


def upload_results(
    local_dir=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    remote_dir=REMOTE_DIR,
):
    '''Upload results to Hugging Face. Args: local_dir (str|Path), repo_id (str), repo_type (str), remote_dir (str). Returns: str.'''
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"results directory not found: {local_dir}")
    return upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        path_in_repo=remote_dir,
    )


if __name__ == "__main__":
    authenticate_hf(token=HF_TOKEN)
    commit_url = upload_results()
    print("Uploaded to:", commit_url)
