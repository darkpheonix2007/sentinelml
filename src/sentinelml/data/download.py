import hashlib
import urllib.request
import zipfile
from pathlib import Path


URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)

RAW_DIRECTORY = Path("data/raw")
ZIP_PATH = RAW_DIRECTORY / "credit_default.zip"

# Leave this empty for the first run.
EXPECTED_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    """Download, verify, and extract the dataset."""

    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print(f"Downloading dataset from {URL}...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print(f"Saved archive to {ZIP_PATH}.")
    else:
        print(f"Using existing archive at {ZIP_PATH}.")

    actual_sha256 = calculate_sha256(ZIP_PATH)
    print(f"Actual SHA256: {actual_sha256}")

    if EXPECTED_SHA256 and actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            "Checksum mismatch! The downloaded file differs "
            "from the expected checksum."
        )

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        print("Archive contents:")

        for filename in archive.namelist():
            print(f"- {filename}")

        archive.extractall(RAW_DIRECTORY)

    print(f"Extracted files to {RAW_DIRECTORY}.")


if __name__ == "__main__":
    main()
