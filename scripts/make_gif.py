"""Assemble frames recorded by web/tests/record.mjs into a compact, evenly timed GIF.

python scripts/make_gif.py <frames-dir> <out.gif> [--fps 12] [--width 880] [--colors 128]

Frames are named <index>_<ms>.png; the GIF resamples them onto a fixed clock (nearest earlier frame), shares one
adaptive palette across all frames (no flicker between frames) and lets Pillow store only changed regions.
"""

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--width", type=int, default=880)
    parser.add_argument("--colors", type=int, default=256)
    parser.add_argument("--skip", type=float, default=0.0, help="seconds to drop from the start")
    args = parser.parse_args()

    files = sorted(args.frames.glob("*.png"))
    stamps = [int(f.stem.split("_")[1]) for f in files]
    if not files:
        raise SystemExit("no frames")
    step = 1000 / args.fps
    start = args.skip * 1000
    times = [start + i * step for i in range(int((stamps[-1] - start) / step) + 1)]

    def frame_at(t: float) -> Path:
        best = files[0]
        for f, s in zip(files, stamps, strict=True):
            if s <= t:
                best = f
            else:
                break
        return best

    picked = [frame_at(t) for t in times]
    first = Image.open(picked[0]).convert("RGB")
    height = round(first.height * args.width / first.width)
    resized = [Image.open(p).convert("RGB").resize((args.width, height), Image.LANCZOS) for p in picked]

    # One palette from a sample of frames, applied to all: stable colours, smaller file.
    picks = resized[:: max(1, len(resized) // 12)][:12]
    sample = Image.new("RGB", (args.width, height * len(picks)))
    for i, im in enumerate(picks):
        sample.paste(im, (0, i * height))
    palette = sample.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in resized]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=round(step),
        loop=0,
        optimize=True,
        disposal=1,
    )
    size = args.out.stat().st_size / 1e6
    print(f"{args.out}: {len(frames)} frames at {args.fps:g} fps, {args.width}x{height}, {size:.1f} MB")


if __name__ == "__main__":
    main()
