from __future__ import annotations

"""Build a paired (track + video-crop) dataset from raw cytoplasmic recordings.

For each recording, the ROCS channel is loaded frame-by-frame (no full video
in RAM) and 25×25-pixel crops are extracted around each particle detection
using the physical-to-pixel conversion from the SPT viewer (px = x / spacing).

All multi-channel TIFs in this dataset use C-major page ordering:
  page[t] = channel 0 (ROCS), frame t

Output
------
video_tracks.csv  — same schema as tracks.csv, plus a ``crop_idx`` column
                    that indexes into the companion .npy file.
video_crops.npy   — float32 array of shape (N, crop_size, crop_size), one
                    crop per row of the CSV.
"""
import pathlib
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import tifffile

from trajan.data import TracksDataFrame, parse_particle_tree

_SPACING = 0.036   # default µm / pixel (matches SPT viewer constant)
_CROP_SIZE = 24    # output crop side length in pixels

CLASS_TO_TYPE: dict[str, str] = {
    "WAVE7mCherry_late_endosome": "endosome",
    "WAVE18mCherry_Golgi": "Golgi",
    "WAVE27mCherry_Postgolgitoendosome": "Postgolgitoendosome",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_tif_meta(tif: tifffile.TiffFile) -> tuple[int, int, float]:
    """Return (n_frames, n_channels, pixel_spacing) from an open TiffFile."""
    spacing = _SPACING
    if tif.is_imagej and tif.imagej_metadata:
        spacing = float(tif.imagej_metadata.get("spacing", _SPACING))

    series = tif.series[0] if tif.series else None
    if series is not None:
        axes = series.axes
        shape = series.shape
        n_frames = shape[axes.index("T")] if "T" in axes else 1
        n_channels = shape[axes.index("C")] if "C" in axes else 1
    else:
        n_frames = len(tif.pages)
        n_channels = 1

    return n_frames, n_channels, spacing


def _load_rocs_frame(tif: tifffile.TiffFile, frame_idx: int, n_frames: int, n_channels: int) -> np.ndarray:
    """Load the ROCS channel at frame_idx from an open TiffFile.

    Pages are stored in C-major order: page[c * n_frames + t].
    ROCS is channel 0 for single-channel files and channel 1 for multi-channel.
    """
    rocs_channel = 1 if n_channels > 1 else 0
    page_idx = frame_idx * n_channels + rocs_channel  # T-major: frames interleave channels
    return tif.pages[page_idx].asarray().astype(np.float32)


def _find_tif(recording_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Return the best candidate TIF for crop extraction.

    Priority:
      1. ROCS.tif  (single-channel ROCS, exact match)
      2. Any *.tif whose name starts with "ROCS"
      3. First *.tif alphabetically
    """
    rocs = recording_dir / "ROCS.tif"
    if rocs.exists():
        return rocs
    tifs = sorted(recording_dir.glob("*.tif"))
    for tif in tifs:
        if tif.name.upper().startswith("ROCS"):
            return tif
    return tifs[0] if tifs else None


def _find_xml(recording_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Return Tracks.xml if present, otherwise the first .xml file found."""
    tracks = recording_dir / "Tracks.xml"
    if tracks.exists():
        return tracks
    xmls = sorted(recording_dir.glob("*.xml"))
    return xmls[0] if xmls else None


def _find_recording_dirs(class_dir: pathlib.Path) -> list[pathlib.Path]:
    """Collect all recording directories for a class.

    Handles two layouts:
      CLASS_DIR/<id>/           — standard numbered recording
      CLASS_DIR/New tracks/<id>/ — additional recordings in a subdirectory
    """
    dirs: list[pathlib.Path] = []
    for subdir in sorted(class_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        if subdir.name == "New tracks":
            for nested in sorted(subdir.iterdir()):
                if nested.is_dir() and not nested.name.startswith("."):
                    dirs.append(nested)
        else:
            dirs.append(subdir)
    return dirs


def _extract_crop(
    frame: np.ndarray,
    py: int,
    px: int,
    size: int,
) -> Optional[np.ndarray]:
    """Extract a size×size patch centred at (py, px), edge-padded at boundaries.

    For even sizes the centre pixel sits at the top-left of the middle 2×2 block.
    Returns None if the centre pixel is entirely outside the frame.
    """
    H, W = frame.shape
    half = size // 2
    r0, r1 = py - half, py - half + size
    c0, c1 = px - half, px - half + size

    pad_top  = max(0, -r0);    r0 = max(0, r0)
    pad_bot  = max(0, r1 - H); r1 = min(H, r1)
    pad_left = max(0, -c0);    c0 = max(0, c0)
    pad_right = max(0, c1 - W); c1 = min(W, c1)

    if r0 >= r1 or c0 >= c1:
        return None

    patch = frame[r0:r1, c0:c1]
    if pad_top or pad_bot or pad_left or pad_right:
        patch = np.pad(
            patch,
            ((pad_top, pad_bot), (pad_left, pad_right)),
            mode="edge",
        )
    return patch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_video_crops_dataset(
    base_dir: pathlib.Path,
    crop_size: int = _CROP_SIZE,
    frame_rate: Optional[float] = None,
    verbose: bool = True,
) -> tuple[TracksDataFrame, np.ndarray]:
    """Build a track + video-crop dataset from all cytoplasmic recordings.

    Walks ``base_dir`` looking for the three class directories defined in
    ``CLASS_TO_TYPE``. For each recording, opens the TIF and reads only the
    frames needed (one at a time) rather than loading the full video. Crops
    are extracted from channel 0 (ROCS) using C-major page indexing.

    Parameters
    ----------
    base_dir : pathlib.Path
        Root directory containing the three class sub-directories.
    crop_size : int
        Side length of each square crop in pixels. Default 24 → 24×24 crops.
    frame_rate : float, optional
        Recording frame rate in Hz. Stored in the returned ``TracksDataFrame``
        and persisted when saving with :meth:`TracksDataFrame.save_npz`.
    verbose : bool
        Print one line per recording showing counts.

    Returns
    -------
    df : TracksDataFrame
        Columns: x, y, frame, label, set, type, crop_idx.
        ``crop_idx`` is the row index into ``crops``.
    crops : np.ndarray
        Shape ``(N, crop_size, crop_size)``, dtype float32.
        Raw pixel values from channel 0 of the ROCS video.
    """
    rows: list[dict] = []
    crops: list[np.ndarray] = []
    set_id = 0

    for class_dir_name, particle_type in CLASS_TO_TYPE.items():
        class_dir = base_dir / class_dir_name
        if not class_dir.is_dir():
            if verbose:
                print(f"[warn] class directory not found: {class_dir}")
            continue

        if verbose:
            print(f"\n{class_dir_name}  ({particle_type})")

        for rec_dir in _find_recording_dirs(class_dir):
            tif_path = _find_tif(rec_dir)
            xml_path = _find_xml(rec_dir)

            rel = rec_dir.relative_to(base_dir)
            if tif_path is None or xml_path is None:
                missing = "TIF" if tif_path is None else "XML"
                if verbose:
                    print(f"  skip {rel}: no {missing}")
                continue

            try:
                particles = parse_particle_tree(ET.parse(xml_path))
            except Exception as exc:
                if verbose:
                    print(f"  error parsing XML {rel}: {exc}")
                continue

            if not particles:
                if verbose:
                    print(f"  {str(rel):<50s}  set={set_id:3d}  particles=   0  dets=    0")
                set_id += 1
                continue

            # Collect the unique frame indices needed to avoid redundant disk reads.
            needed: dict[int, list[tuple[int, float, float, int]]] = {}
            for label_idx, particle in enumerate(particles):
                for t, x, y in particle:
                    frame_idx = int(t)
                    needed.setdefault(frame_idx, []).append((label_idx, x, y))

            n_skipped = 0
            det_before = len(crops)

            try:
                with tifffile.TiffFile(str(tif_path)) as tif:
                    n_frames, n_channels, spacing = _get_tif_meta(tif)

                    for frame_idx in sorted(needed.keys()):
                        if frame_idx >= n_frames:
                            n_skipped += len(needed[frame_idx])
                            continue

                        frame_ch0 = _load_rocs_frame(tif, frame_idx, n_frames, n_channels)

                        for label_idx, x, y in needed[frame_idx]:
                            px = int(round(x / spacing))
                            py = int(round(y / spacing))
                            crop = _extract_crop(frame_ch0, py, px, crop_size)
                            if crop is None:
                                n_skipped += 1
                                continue
                            crops.append(crop)
                            rows.append(
                                {
                                    "x": x,
                                    "y": y,
                                    "frame": frame_idx,
                                    "label": label_idx,
                                    "set": set_id,
                                    "type": particle_type,
                                    "crop_idx": len(crops) - 1,
                                }
                            )
            except Exception as exc:
                if verbose:
                    print(f"  error reading TIF {rel}: {exc}")
                continue

            n_dets = len(crops) - det_before
            if verbose:
                skip_str = f"  ({n_skipped} skipped)" if n_skipped else ""
                print(
                    f"  {str(rel):<50s}  set={set_id:3d}  "
                    f"particles={len(particles):4d}  dets={n_dets:5d}{skip_str}"
                )
            set_id += 1

    df = TracksDataFrame(rows, frame_rate=frame_rate)
    if crops:
        raw = np.stack(crops).astype(np.float32)
        # Sort to match tracks.csv ordering: set → label → frame
        df = df.sort_values(["set", "label", "frame"]).reset_index(drop=True)
        crops_arr = raw[df["crop_idx"].values]
        df["crop_idx"] = np.arange(len(df))
    else:
        crops_arr = np.zeros((0, crop_size, crop_size), dtype=np.float32)

    return df, crops_arr
