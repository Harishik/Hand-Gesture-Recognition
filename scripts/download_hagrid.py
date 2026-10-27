"""Download and extract the HaGRIDv2 annotations (boxes, labels, user ids and MediaPipe hand landmarks).

The annotations alone (~720 MB zipped) are enough to train landmark-based classifiers; the 1.5 TB of images
are not needed for that.

    python scripts/download_hagrid.py [--out data/raw/hagrid]
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

ANNOTATIONS_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
REPO_ROOT = Path(__file__).resolve().parents[1]


def download(url, dest):
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}\n  -> {dest}")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    tmp.rename(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "raw" / "hagrid")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "annotations.zip"
    download(ANNOTATIONS_URL, archive)

    print(f"Extracting to {args.out}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
