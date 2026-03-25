"""
Data Download Script for EdgeHAR.

Downloads the UCI Human Activity Recognition dataset from the UCI Machine
Learning Repository, extracts it, and verifies the downloaded files.

Usage:
    python data/download_data.py
"""

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG = {
    "url": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00240/UCI%20HAR%20Dataset.zip"
    ),
    "data_dir": Path(__file__).resolve().parent,
    "zip_filename": "UCI_HAR_Dataset.zip",
    "extract_dirname": "UCI_HAR_Dataset",
}


def download_file(url: str, dest: Path) -> None:
    """Download a file from a URL with a progress bar.

    Args:
        url: The URL to download from.
        dest: The destination file path.

    Raises:
        requests.HTTPError: If the download request fails.
    """
    print(f"📥 Downloading from:\n   {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192

    with (
        open(dest, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar,
    ):
        for chunk in response.iter_content(chunk_size=block_size):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"✅ Downloaded to: {dest}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file to a target directory.

    Args:
        zip_path: Path to the zip file.
        extract_to: Directory to extract into.

    Raises:
        zipfile.BadZipFile: If the zip file is corrupted.
    """
    print(f"📦 Extracting to: {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Extraction complete!")


def verify_dataset(data_dir: Path) -> None:
    """Verify that the UCI HAR Dataset was downloaded and extracted correctly.

    Args:
        data_dir: Path to the data directory containing UCI HAR Dataset.

    Raises:
        FileNotFoundError: If required dataset files are missing.
    """
    # The UCI zip extracts to "UCI HAR Dataset" (with spaces)
    # We check both possible directory names
    possible_names = ["UCI HAR Dataset", "UCI_HAR_Dataset"]
    dataset_dir = None

    for name in possible_names:
        candidate = data_dir / name
        if candidate.exists():
            dataset_dir = candidate
            break

    if dataset_dir is None:
        raise FileNotFoundError(
            f"❌ Dataset directory not found in {data_dir}. "
            f"Expected one of: {possible_names}"
        )

    required_dirs = ["train", "test"]
    required_files = [
        "train/X_train.txt",
        "train/y_train.txt",
        "test/X_test.txt",
        "test/y_test.txt",
        "activity_labels.txt",
    ]

    print("\n📋 Verifying dataset structure...")

    for d in required_dirs:
        dir_path = dataset_dir / d
        if not dir_path.is_dir():
            raise FileNotFoundError(f"❌ Missing directory: {dir_path}")
        print(f"   ✅ Found directory: {d}/")

    for f in required_files:
        file_path = dataset_dir / f
        if not file_path.is_file():
            raise FileNotFoundError(f"❌ Missing file: {file_path}")
        print(f"   ✅ Found file: {f}")

    # Check inertial signals
    for split in ["train", "test"]:
        inertial_dir = dataset_dir / split / "Inertial Signals"
        if not inertial_dir.is_dir():
            raise FileNotFoundError(
                f"❌ Missing Inertial Signals directory: {inertial_dir}"
            )

        signal_files = list(inertial_dir.glob("*.txt"))
        print(f"   ✅ Found {len(signal_files)} signal files in {split}/Inertial Signals/")

    print(f"\n🎉 Dataset verified successfully at: {dataset_dir}")
    print(f"   Total files: {sum(1 for _ in dataset_dir.rglob('*') if _.is_file())}")


def main() -> None:
    """Main function to download and setup the UCI HAR Dataset."""
    data_dir = CONFIG["data_dir"]
    zip_path = data_dir / CONFIG["zip_filename"]
    extract_dir = data_dir / CONFIG["extract_dirname"]

    # Check if already downloaded
    if extract_dir.exists() or (data_dir / "UCI HAR Dataset").exists():
        print("📂 Dataset already exists. Verifying...")
        verify_dataset(data_dir)
        return

    # Create data directory if needed
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download
        download_file(CONFIG["url"], zip_path)

        # Extract
        extract_zip(zip_path, data_dir)

        # Verify
        verify_dataset(data_dir)

        # Clean up zip file
        zip_path.unlink()
        print(f"🗑️  Removed zip file: {zip_path}")

    except requests.RequestException as e:
        print(f"❌ Download failed: {e}")
        print("   Please check your internet connection and try again.")
        raise
    except zipfile.BadZipFile as e:
        print(f"❌ Zip file is corrupted: {e}")
        if zip_path.exists():
            zip_path.unlink()
        raise


if __name__ == "__main__":
    main()
