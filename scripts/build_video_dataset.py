"""Build the video-crop dataset from all cytoplasmic recordings.

Produces two files in data/cytoplasmic/:
  video_tracks.csv  — TracksDataFrame-compatible CSV with a crop_idx column
  video_crops.npy   — float32 array of shape (N, 25, 25), one crop per row

Usage
-----
    python scripts/build_video_dataset.py
    python scripts/build_video_dataset.py --base data/cytoplasmic --crop-half 12
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trajan.video_crops import build_video_crops_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="data/cytoplasmic", help="root data directory")
    parser.add_argument("--crop-size", type=int, default=24, help="side length of each crop in pixels (default 24 → 24×24)")
    args = parser.parse_args()

    base_dir = pathlib.Path(args.base)
    if not base_dir.is_dir():
        sys.exit(f"Directory not found: {base_dir}")

    print(f"Building video crop dataset from {base_dir} ...")
    df, crops = build_video_crops_dataset(base_dir, crop_size=args.crop_size)

    if df.empty:
        sys.exit("No detections found — check that TIF and XML files are present.")

    out_csv = base_dir / "video_tracks.csv"
    out_npy = base_dir / "video_crops.npy"

    df.to_csv(out_csv, index=False)
    np.save(out_npy, crops)

    print(f"\nSaved {len(df):,} detections across {df['set'].nunique()} recordings")
    print(f"  crops shape : {crops.shape}  dtype: {crops.dtype}")
    print(f"  {out_csv}")
    print(f"  {out_npy}")

    # Brief per-class summary
    print("\nDetections per class:")
    for ptype, grp in df.groupby("type"):
        print(f"  {ptype:<30s}  {len(grp):6,d}  ({grp['set'].nunique()} recordings)")


if __name__ == "__main__":
    main()
