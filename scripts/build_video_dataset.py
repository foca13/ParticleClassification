"""Build the video-crop dataset from all cytoplasmic recordings.

Produces one file in data/cytoplasmic/video_crops/:
  video_crops_{crop_size}.npz  — compressed archive containing tracks CSV,
                                  frame_rate, and raw float32 crops of shape
                                  (N, crop_size, crop_size)

Usage
-----
    python scripts/build_video_dataset.py
    python scripts/build_video_dataset.py --base data/cytoplasmic --crop-size 32 --frame-rate 10
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trajan.video_crops import build_video_crops_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="data/cytoplasmic", help="root data directory")
    parser.add_argument("--crop_size", type=int, default=24, help="side length of each crop in pixels (default 24 → 24×24)")
    parser.add_argument("--frame_rate", type=float, default=None, help="recording frame rate in Hz (stored in the output npz)")
    args = parser.parse_args()

    base_dir = pathlib.Path(args.base)
    if not base_dir.is_dir():
        sys.exit(f"Directory not found: {base_dir}")

    print(f"Building video crop dataset from {base_dir} ...")
    df, crops = build_video_crops_dataset(base_dir, crop_size=args.crop_size, frame_rate=args.frame_rate)

    if df.empty:
        sys.exit("No detections found — check that TIF and XML files are present.")

    out_path = base_dir / "video_crops" / f"video_crops_{args.crop_size}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.save_npz(out_path, crops=crops)

    print(f"\nSaved {len(df):,} detections across {df['set'].nunique()} recordings")
    if args.frame_rate is not None:
        print(f"  frame_rate  : {args.frame_rate} Hz")
    print(f"  crops shape : {crops.shape}  dtype: {crops.dtype}")
    print(f"  {out_path}")

    # Brief per-class summary
    print("\nDetections per class:")
    for ptype, grp in df.groupby("type"):
        print(f"  {ptype:<30s}  {len(grp):6,d}  ({grp['set'].nunique()} recordings)")


if __name__ == "__main__":
    main()
