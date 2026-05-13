import io
import pathlib
import warnings
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from trajan.features import compute_trajectory_features


def load_xml_files(dir_path: str) -> List[ET.ElementTree]:
    """Recursively load all XML files from a directory.

    Parameters
    ----------
    dir_path : str
        Path to the directory to search for XML files.

    Returns
    -------
    List[ET.ElementTree]
        A list of parsed XML trees, one per file found.
    """
    dir_obj = pathlib.Path(dir_path)
    xml_files = []
    for file in dir_obj.rglob("*.xml"):
        tree = ET.parse(file)
        xml_files.append(tree)
    return xml_files


def parse_particle_tree(particle_tree: ET.ElementTree) -> List[np.ndarray]:
    """Parse a single XML tree into a list of particle trajectories.

    Expects an XML structure with a 'Tracks' root element containing
    'particle' elements, each with 'detection' children. Example:

        <Tracks nTracks="2">
            <particle>
                <detection t="0" x="1.0" y="2.0"/>
                <detection t="1" x="1.5" y="2.3"/>
            </particle>
            <particle>
                ...
            </particle>
        </Tracks>

    Parameters
    ----------
    particle_tree : ET.ElementTree
        A parsed XML tree following the structure described above.

    Returns
    -------
    List[np.ndarray]
        A list of trajectories, one per particle. Each trajectory is a
        list of (t, x, y) tuples representing detections in temporal order.

    Raises
    ------
    AssertionError
        If the number of parsed particles does not match the nTracks
        attribute declared in the 'Tracks' element.
    """
    n_tracks = 0
    n_particles = 0
    particles = []
    for element in particle_tree.iter():
        if element.tag == "Tracks":
            n_tracks = int(element.get("nTracks"))
        elif element.tag == "particle":
            n_particles += 1
            particles.append([])
        elif element.tag == "detection":
            timestep = int(element.get("t"))
            x = float(element.get("x"))
            y = float(element.get("y"))
            particles[n_particles - 1].append((timestep, x, y))
    assert n_tracks == n_particles, "number of tracks should match number of particles"
    return particles


def parse_particle_xml_files(
    xml_files: List[ET.ElementTree],
) -> List[List[np.ndarray]]:
    """Parse a list of XML trees, discarding empty recordings.

    Parameters
    ----------
    xml_files : List[ET.ElementTree]
        A list of parsed XML trees, as returned by load_xml_files.

    Returns
    -------
    List[List[np.ndarray]]
        A list of recordings, each being a list of particle trajectories
        as returned by parse_particle_tree. Empty recordings are excluded.
    """
    tracks = []
    for file in xml_files:
        particles = parse_particle_tree(file)
        if len(particles) > 0:
            tracks.append(particles)
    return tracks


class TracksDataFrame(pd.DataFrame):
    """A DataFrame for storing and manipulating particle tracking data.

    Each row represents a single particle detection at a specific frame.
    Detections belonging to the same particle across frames are linked by a
    shared label, forming a trajectory. Multiple particles can be recorded
    across multiple videos, identified by the 'set' column.

    Extends pd.DataFrame, so all standard pandas operations are available.
    Operations that return a new DataFrame will return a TracksDataFrame
    instance, preserving the frame_rate metadata.

    Parameters
    ----------
    frame_rate : float, optional
        The frame rate of the recording in frames per second. Used for
        time-aware descriptions of the tracks.
    *args, **kwargs
        Passed to pd.DataFrame.

    Columns
    -------
    set : str or int
        Identifier for the video/recording session.
    frame : int
        Frame index of the detection.
    label : int
        Particle identifier. Detections sharing a label belong to the
        same trajectory.
    x : float
        X coordinate of the detection centroid.
    y : float
        Y coordinate of the detection centroid.

    Methods
    -------
    describe_tracks()
        Prints and returns a summary of the tracks grouped by particle type
        and recording, including track counts and average track length.
    split_train_test(test_size, seed)
        Splits the data into train and test sets, stratified by particle
        type and video.
    compute_increments()
        Computes per-step increment magnitudes across all trajectories,
        handling gaps in frame continuity. Returns velocity magnitudes
        if frame_rate is set, otherwise raw displacement magnitudes.

    Examples
    --------
    >>> df = TracksDataFrame(raw_df, frame_rate=30)
    >>> train, test = df.split_train_test(test_size=0.2, seed=42)
    >>> increments = train.compute_increments()
    """

    _metadata = ["frame_rate"]

    def __init__(self, *args, frame_rate=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frame_rate = frame_rate

    @property
    def _constructor(self) -> "TracksDataFrame":
        return TracksDataFrame

    @classmethod
    def concat(cls, dfs: List["TracksDataFrame"], frame_rate: Optional[float] = None) -> "TracksDataFrame":
        """Concatenate multiple TracksDataFrame objects into one.

        Parameters
        ----------
        dfs : List[TracksDataFrame]
            DataFrames to concatenate.
        frame_rate : float, optional
            Frame rate for the result. Defaults to the first df's frame_rate.
        """
        if frame_rate is None and dfs:
            frame_rate = dfs[0].frame_rate
        adjusted = []
        offset = 0
        for df in dfs:
            copy = df.copy()
            copy["set"] = copy["set"] + offset
            offset += int(df["set"].max()) + 1
            adjusted.append(copy)
        return cls(pd.concat(adjusted, ignore_index=True), frame_rate=frame_rate)

    def describe_tracks(self, print_msg: bool = True) -> Dict:
        particle_types = self["type"].unique().tolist()
        description = {"particle_types": particle_types}

        if self.frame_rate is not None:
            description["frame_rate"] = self.frame_rate

        for particle_type in particle_types:
            selected_type = self[self["type"] == particle_type]
            videos = selected_type["set"].unique()

            total_tracks = selected_type.groupby("set")["label"].nunique().sum()
            avg_track_len = np.round(len(selected_type) / total_tracks, 2)

            description[particle_type] = {
                "n_recordings": len(videos),
                "recording_ids": list(map(int, videos)),
                "n_tracks": total_tracks,
                "avg_track_len": avg_track_len,
            }

        if print_msg:
            for key, value in description.items():
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")
        return description

    def track_lengths(self) -> Dict[str, Dict[int, Dict[int, int]]]:
        """Return the length of every trajectory, grouped by type, recording, and label.

        Returns
        -------
        dict
            ``{particle_type: {recording_id: {label: length}}}``
            where *length* is the number of detections in that trajectory.
        """
        result: Dict[str, Dict[int, Dict[int, int]]] = {}
        for ptype in self["type"].unique():
            result[ptype] = {}
            subset = self[self["type"] == ptype]
            for rec in sorted(subset["set"].unique()):
                rec_data = subset[subset["set"] == rec]
                result[ptype][int(rec)] = {
                    int(lbl): int(len(grp))
                    for lbl, grp in rec_data.groupby("label")
                }
        return result

    def split_train_test(self, test_size: float = 0.25, seed: Optional[int] = None) -> Tuple["TracksDataFrame", "TracksDataFrame"]:
        """Split data into train and test sets by recording.

        Splitting is stratified by particle type, so each particle type
        contributes test recordings proportionally. All tracks from a given
        recording appear in either train or test, never both, simulating
        generalisation to unseen recordings.

        Parameters
        ----------
        test_size : float, optional
            Fraction of recordings to use for testing. Default is 0.25.
        seed : int, optional
            Random seed for reproducibility. Default is None.

        Returns
        -------
        Tuple[TracksDataFrame, TracksDataFrame]
            Train and test sets as TracksDataFrame instances.
        """
        rng = np.random.default_rng(seed)
        train_data, test_data = [], []

        for particle_type in self["type"].unique():
            selected_type = self[self["type"] == particle_type]
            recordings = selected_type["set"].unique()
            n_test = int(round(len(recordings) * test_size))
            if n_test == 0:
                warnings.warn(f"No test recordings for particle type '{particle_type}'")
            test_recordings = rng.choice(recordings, n_test, replace=False)
            test_data.append(selected_type[selected_type["set"].isin(test_recordings)])
            train_data.append(selected_type[~selected_type["set"].isin(test_recordings)])

        train_data = TracksDataFrame(pd.concat(train_data, ignore_index=True), frame_rate=self.frame_rate)
        test_data = TracksDataFrame(pd.concat(test_data, ignore_index=True), frame_rate=self.frame_rate)

        return train_data, test_data

    def split_train_test_manual(
        self, val_sets: Dict[str, List[int]]
    ) -> Tuple["TracksDataFrame", "TracksDataFrame"]:
        """Split data into train and validation sets using explicit recording indices.

        Parameters
        ----------
        val_sets : Dict[str, List[int]]
            Mapping from particle type to a list of set (recording) indices
            to hold out for validation. Types absent from the dict are
            assigned entirely to the training set with a warning.

        Returns
        -------
        Tuple[TracksDataFrame, TracksDataFrame]
            Train and validation sets as TracksDataFrame instances.
        """
        train_data, val_data = [], []

        for particle_type in self["type"].unique():
            selected_type = self[self["type"] == particle_type]
            val_set_ids = val_sets.get(particle_type, [])
            if not val_set_ids:
                warnings.warn(
                    f"No validation sets specified for particle type '{particle_type}', "
                    "all its recordings go to train."
                )
            val_mask = selected_type["set"].isin(val_set_ids)
            val_data.append(selected_type[val_mask])
            train_data.append(selected_type[~val_mask])

        train_data = TracksDataFrame(pd.concat(train_data, ignore_index=True), frame_rate=self.frame_rate)
        val_data = TracksDataFrame(pd.concat(val_data, ignore_index=True), frame_rate=self.frame_rate)

        return train_data, val_data

    def compute_increments(self) -> np.ndarray:
        """Compute per-step increment magnitudes across all trajectories.

        Handles gaps in frame continuity by splitting trajectories at
        missing frames. If ``frame_rate`` is set, magnitudes are divided
        by the inter-frame time to give speed magnitudes (spatial units /
        second); otherwise raw displacement magnitudes are returned.

        Returns
        -------
        np.ndarray
            1D array of increment magnitudes, one per consecutive detection
            pair across all trajectories and videos.
        """
        increments = []

        for video in self["set"].unique():
            video_data = self[self["set"] == video]
            for particle in video_data["label"].unique():
                p = video_data[video_data["label"] == particle].sort_values("frame")
                frames_arr = p["frame"].to_numpy()
                coords = p[["x", "y"]].to_numpy()
                gaps = np.where(np.diff(frames_arr) > 1)[0]
                frame_sections = np.split(frames_arr, gaps + 1)
                coord_sections = np.split(coords, gaps + 1)
                for frames_sec, coords_sec in zip(frame_sections, coord_sections):
                    if len(coords_sec) < 2:
                        continue
                    diffs = np.diff(coords_sec, axis=0)
                    magnitudes = np.linalg.norm(diffs, axis=1)
                    if self.frame_rate is not None:
                        dt_seconds = np.diff(frames_sec) / self.frame_rate
                        magnitudes = magnitudes / dt_seconds
                    increments.append(magnitudes)

        return np.concatenate(increments) if increments else np.array([], dtype=np.float64)

    def compute_features(self) -> pd.DataFrame:
        """Compute hand-crafted trajectory features for all particles.
        ...
        """
        rows = []
        for video in self["set"].unique():
            df_video = self[self["set"] == video]
            for label in df_video["label"].unique():
                track = df_video[df_video["label"] == label].sort_values("frame")
                coords = track[["x", "y"]].to_numpy()
                if len(coords) < 3:
                    continue
                features = compute_trajectory_features(coords, frame_rate=self.frame_rate)
                features["set"] = video
                features["label"] = label
                if "type" in self.columns:
                    features["type"] = track["type"].iloc[0]
                rows.append(features)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Serialisation — npz (default) and HDF5 (optional, requires h5py)
    # ------------------------------------------------------------------

    def save_npz(self, path: Union[str, pathlib.Path], crops: Optional[np.ndarray] = None) -> None:
        """Save tracks and (optionally) crops to a single ``.npz`` file.

        Parameters
        ----------
        path : str or Path
            Destination file. The ``.npz`` extension is added if absent.
        crops : np.ndarray, optional
            Crop array of shape ``(N, C, H, W)`` or ``(N, H, W)``.
            Omit if this dataset has no associated crops.

        Examples
        --------
        >>> df.save_npz("data/cytoplasmic/dataset.npz", crops=crops_array)
        >>> df.save_npz("data/cytoplasmic/tracks_only.npz")
        """
        buf = io.StringIO()
        self.to_csv(buf, index=False)
        csv_bytes = np.frombuffer(buf.getvalue().encode(), dtype=np.uint8)

        arrays: dict = {"tracks_csv": csv_bytes}
        if self.frame_rate is not None:
            arrays["frame_rate"] = np.array([self.frame_rate])
        if crops is not None:
            arrays["crops"] = crops

        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(cls, path: Union[str, pathlib.Path]) -> Tuple["TracksDataFrame", Optional[np.ndarray]]:
        """Load tracks and crops from a ``.npz`` file saved by :meth:`save_npz`.

        Parameters
        ----------
        path : str or Path
            Path to the ``.npz`` file.

        Returns
        -------
        tracks : TracksDataFrame
        crops : np.ndarray or None
            ``None`` when the file contains no crops.

        Examples
        --------
        >>> tracks, crops = TracksDataFrame.load_npz("data/cytoplasmic/dataset.npz")
        """
        data = np.load(path, allow_pickle=False)
        csv_str = data["tracks_csv"].tobytes().decode()
        frame_rate = float(data["frame_rate"][0]) if "frame_rate" in data else None
        crops = data["crops"] if "crops" in data else None
        df = cls(pd.read_csv(io.StringIO(csv_str)), frame_rate=frame_rate)
        return df, crops

    def save_hdf5(self, path: Union[str, pathlib.Path], crops: Optional[np.ndarray] = None) -> None:
        """Save tracks and (optionally) crops to a single HDF5 file.

        Requires ``h5py``.  The tracks are stored under the key ``"tracks"``
        using pandas' built-in HDF5 support; crops are stored as a dataset
        under ``"crops"``.  Frame rate is stored as a root-level attribute.

        Parameters
        ----------
        path : str or Path
            Destination ``.h5`` file.
        crops : np.ndarray, optional
            Crop array of shape ``(N, C, H, W)`` or ``(N, H, W)``.

        Examples
        --------
        >>> df.save_hdf5("data/cytoplasmic/dataset.h5", crops=crops_array)
        """
        import h5py  # optional dependency

        path = pathlib.Path(path)
        pd.DataFrame(self).to_hdf(path, key="tracks", mode="w")

        with h5py.File(path, "a") as f:
            if self.frame_rate is not None:
                f.attrs["frame_rate"] = self.frame_rate
            if crops is not None:
                f.create_dataset("crops", data=crops, compression="gzip")

    @classmethod
    def load_hdf5(cls, path: Union[str, pathlib.Path]) -> Tuple["TracksDataFrame", Optional[np.ndarray]]:
        """Load tracks and crops from an HDF5 file saved by :meth:`save_hdf5`.

        Requires ``h5py``.

        Parameters
        ----------
        path : str or Path
            Path to the ``.h5`` file.

        Returns
        -------
        tracks : TracksDataFrame
        crops : np.ndarray or None
            ``None`` when the file contains no crops dataset.

        Examples
        --------
        >>> tracks, crops = TracksDataFrame.load_hdf5("data/cytoplasmic/dataset.h5")
        """
        import h5py  # optional dependency

        df = pd.read_hdf(path, key="tracks")
        with h5py.File(path, "r") as f:
            frame_rate = float(f.attrs["frame_rate"]) if "frame_rate" in f.attrs else None
            crops = f["crops"][:] if "crops" in f else None

        return cls(df, frame_rate=frame_rate), crops


def to_tracks_dataframe(
    track_recordings: List[List[np.ndarray]],
    frame_rate: Optional[float] = None,
    particle_type: Optional[Union[str, List[str]]] = None,
) -> TracksDataFrame:
    """Convert parsed XML recordings into a TracksDataFrame.

    Parameters
    ----------
    track_recordings : List[List[np.ndarray]]
        A list of recordings as returned by parse_particle_xml_files.
        Each recording is a list of trajectories (one per track), each
        being an array of (t, x, y) tuples.
    frame_rate : float, optional
        Frame rate of the recordings in frames per second. Default is None.
    particle_type : str or list of str, optional
        Particle type label(s) for the "type" column. If a single string,
        all rows share that type. If a list, its length must match the number
        of recordings and each recording's rows get the corresponding type.
        If not provided, all rows are labelled "unspecified".

    Returns
    -------
    TracksDataFrame
        A TracksDataFrame with columns set, frame, label, type, x, y.
        Labels restart from 0 for each recording.
    """
    if particle_type is None:
        type_per_set = ["unspecified"] * len(track_recordings)
    elif isinstance(particle_type, str):
        type_per_set = [particle_type] * len(track_recordings)
    else:
        if len(particle_type) != len(track_recordings):
            raise ValueError(
                f"particle_type list length ({len(particle_type)}) must match "
                f"number of recordings ({len(track_recordings)})"
            )
        type_per_set = list(particle_type)

    rows = []
    for set_idx, recording in enumerate(track_recordings):
        for label_idx, particle in enumerate(recording):
            for t, x, y in particle:
                rows.append({
                    "x": x,
                    "y": y,
                    "frame": t,
                    "label": label_idx,
                    "set": set_idx,
                    "type": type_per_set[set_idx],
                })

    return TracksDataFrame(pd.DataFrame(rows), frame_rate=frame_rate)
