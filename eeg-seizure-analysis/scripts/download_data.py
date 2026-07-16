"""Download and extract the public Kaggle dataset without Kaggle credentials."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
ARCHIVE = DATA_DIR / "epileptic-seizure-recognition.zip"
URL = "https://www.kaggle.com/api/v1/datasets/download/harunshimanto/epileptic-seizure-recognition"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement depuis {URL}")
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, ARCHIVE.open("wb") as destination:
        shutil.copyfileobj(response, destination)
    with zipfile.ZipFile(ARCHIVE) as zipped:
        zipped.extractall(DATA_DIR)
    print(f"Dataset extrait dans {DATA_DIR}")


if __name__ == "__main__":
    main()

