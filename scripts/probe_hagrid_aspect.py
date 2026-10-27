"""Measure real HaGRID image sizes without downloading images: range-read JPEG headers from a public sample zip.

HaGRIDv2 annotations don't record image size, but landmarks are normalised by width and height separately, so the
aspect ratio is needed to recover true hand shape. This samples image sizes and writes {image_id: [w, h]}.

python scripts/probe_hagrid_aspect.py [--n 400] [--out data/processed/hagrid_aspect_sample.json]
"""

import argparse
import json
import random
import struct
import urllib.request
import zipfile
from pathlib import Path

SAMPLE_ZIP_URL = (
    "https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p/resolve/main/hagrid-sample-30k-384p.zip"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


class HttpRangeFile:
    """Minimal read-only, seekable file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url):
        response = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}))
        self.url = response.geturl()  # follow the redirect once
        self.size = int(response.headers["Content-Range"].split("/")[1])
        self.pos = 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = urllib.request.urlopen(request).read()
        self.pos += len(data)
        return data


def jpeg_size(data):
    """(width, height) from the SOF marker of a JPEG byte prefix, or None if not found."""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        i += 2 + struct.unpack(">H", data[i + 2 : i + 4])[0]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "processed" / "hagrid_aspect_sample.json")
    args = parser.parse_args()

    archive = zipfile.ZipFile(HttpRangeFile(SAMPLE_ZIP_URL))
    images = [name for name in archive.namelist() if name.endswith(".jpg")]
    sizes = {}
    for name in random.Random(args.seed).sample(images, args.n):
        with archive.open(name) as f:
            size = jpeg_size(f.read(4096))
        if size:
            sizes[Path(name).stem] = size

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sizes))
    print(f"Saved {len(sizes)} image sizes to {args.out}")


if __name__ == "__main__":
    main()
