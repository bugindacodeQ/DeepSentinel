"""
Downloads trained DeepSentinel model weights from Hugging Face Hub.

Usage:
    python scripts/download_weights.py --repo YOUR_HF_USERNAME/DeepSentinel

After training on Colab (notebooks/03_training.ipynb), upload your weights:
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj="models/weights/efficientnet_b4.pth",
        path_in_repo="efficientnet_b4.pth",
        repo_id="YOUR_HF_USERNAME/DeepSentinel",
        repo_type="model",
    )
Then run this script to pull them down on any machine.
"""
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download(repo_id: str, filename: str, dest: Path):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} from {repo_id} ...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest.parent),
        local_dir_use_symlinks=False,
    )
    print(f"Saved to: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Download DeepSentinel weights from HF Hub")
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repo ID, e.g. yourname/DeepSentinel",
    )
    parser.add_argument(
        "--filename",
        default="efficientnet_b4.pth",
        help="Filename in the HF repo (default: efficientnet_b4.pth)",
    )
    parser.add_argument(
        "--dest",
        default="models/weights/efficientnet_b4.pth",
        help="Local destination path (default: models/weights/efficientnet_b4.pth)",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    if dest.exists():
        print(f"Weights already exist at {dest}. Delete it first to re-download.")
        sys.exit(0)

    download(repo_id=args.repo, filename=args.filename, dest=dest)
    print("\nDone. You can now run the app:")
    print("  python app/gradio_app.py")
    print("  uvicorn app.api:app --reload")


if __name__ == "__main__":
    main()
