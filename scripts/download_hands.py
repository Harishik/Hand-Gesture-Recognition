"""Download HANDS (Nuzzi et al. 2021, CC BY 4.0) RGB archives and annotations, one subject at a time.

    python scripts/download_hands.py 2 3      # Subject2.zip/.txt, Subject3.zip/.txt into data/raw/hands
"""

import json
import subprocess
import sys
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/ndrczc35bt"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "hands"


def main(subjects):
    OUT.mkdir(parents=True, exist_ok=True)
    files = json.loads(subprocess.run(["curl", "-sSLf", API], check=True, capture_output=True).stdout)["files"]
    urls = {f["filename"]: f["content_details"]["download_url"] for f in files}
    for s in subjects:
        for name in (f"Subject{s}.txt", f"Subject{s}.zip"):
            dest = OUT / name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            print("downloading", name, flush=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            # Mendeley's file host refuses Python's default client (HTTP 403); curl works.
            subprocess.run(["curl", "-sSLf", "--retry", "3", "-o", str(tmp), urls[name]], check=True)
            tmp.replace(dest)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5])
