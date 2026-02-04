from pathlib import Path

from huggingface_hub import get_token, snapshot_download

from hf_utils import authenticate_hf

HF_TOKEN = get_token()
REPO_ID = "Malassila/eb-mia"
REPO_TYPE = "model"
LOCAL_DIR = Path("training")
REMOTE_DIR = "train_splits"


def download_train_splits(
    local_dir=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    remote_dir=REMOTE_DIR,
):
    '''Download train splits from Hugging Face. Args: local_dir (str|Path), repo_id (str), repo_type (str), remote_dir (str). Returns: str.'''
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        allow_patterns=f"{remote_dir}/**",
    )


if __name__ == "__main__":
    authenticate_hf(token=HF_TOKEN)
    local_dir = download_train_splits()
    print("Downloaded to:", local_dir)
