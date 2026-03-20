from pathlib import Path
import argparse

from huggingface_hub import get_token, upload_folder

try:
    from hf_utils import authenticate_hf
except ModuleNotFoundError:
    from hf_scripts.hf_utils import authenticate_hf

HF_TOKEN = get_token()
REPO_ID = "Malassila/eb-mia"
REPO_TYPE = "model"


def upload_directory(
    local_dir,
    remote_dir,
    file_pattern=None,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
):
    '''
    Upload a local directory or matching files to a Hugging Face repo.
    Args:
        local_dir (str | Path): Local directory to upload.
        remote_dir (str): Target directory in the remote repo.
        file_pattern (str | None): Optional upload pattern relative to `local_dir`.
        repo_id (str): Hugging Face repository id.
        repo_type (str): Hugging Face repository type.
    Returns:
        str: Commit URL returned by Hugging Face Hub.
    '''
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"local directory not found: {local_dir}")
    return upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        path_in_repo=remote_dir,
        allow_patterns=file_pattern,
    )


def parse_args():
    '''
    Parse CLI arguments.
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    '''
    parser = argparse.ArgumentParser(description="Upload a local directory to Hugging Face.")
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Local directory to upload.",
    )
    parser.add_argument(
        "--remote-dir",
        required=True,
        help="Target directory path in the Hugging Face repo.",
    )
    parser.add_argument(
        "--file-pattern",
        default=None,
        help="Optional glob pattern (relative to --local-dir) to upload matching files only, e.g. '*-epoch10.pth'.",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"Hugging Face repo id (default: {REPO_ID}).",
    )
    parser.add_argument(
        "--repo-type",
        default=REPO_TYPE,
        help=f"Hugging Face repo type (default: {REPO_TYPE}).",
    )
    return parser.parse_args()


def main():
    '''
    Run the upload CLI.
    Returns:
        None
    '''
    args = parse_args()
    authenticate_hf(token=HF_TOKEN)
    commit_url = upload_directory(
        local_dir=args.local_dir,
        remote_dir=args.remote_dir,
        file_pattern=args.file_pattern,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
    )
    print("Uploaded to:", commit_url)


if __name__ == "__main__":
    main()
