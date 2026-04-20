import json
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd


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
    centroid-0 : float
        Row (y) coordinate of the detection centroid.
    centroid-1 : float
        Column (x) coordinate of the detection centroid.

    Methods
    -------
    describe_tracks()
        Prints and returns a summary of the tracks grouped by particle type
        and recording, including track counts and average track length.
    split_train_test(test_size, seed)
        Splits the data into train and test sets, stratified by particle
        type and video.
    compute_displacements()
        Computes per-step displacement magnitudes across all trajectories,
        handling gaps in frame continuity.

    Examples
    --------
    >>> df = TracksDataFrame(raw_df, frame_rate=30)
    >>> train, test = df.split_train_test(test_size=0.2, seed=42)
    >>> displacements = train.compute_displacements()
    """

    def __init__(self, *args, frame_rate = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frame_rate = frame_rate

    # This ensures operations that return a DataFrame return YOUR class instead
    @property
    def _constructor(self) -> "TracksDataFrame":
        return TracksDataFrame

    def describe_tracks(self) -> Dict:
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
                "n_tracks": total_tracks,
                "avg_track_len": avg_track_len,
            }

        print(json.dumps(description, indent=2, default=str))
        return description

    def split_train_test(self, test_size: float = 0.25, seed=None) -> Tuple["TracksDataFrame", "TracksDataFrame"]:
        """Split data into train and test sets.

        Splitting is stratified by particle type and video, so each
        recording contributes test tracks proportionally.

        Parameters
        ----------
        test_size : float, optional
            Fraction of tracks to use for testing. Default is 0.25.
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
            for video in selected_type["set"].unique():
                tracks = selected_type[selected_type["set"] == video]
                track_ids = tracks["label"].unique()
                n_test = int(round(len(track_ids) * test_size))
                if n_test == 0:
                    warnings.warn(f"No test tracks for particle type '{particle_type}' in video '{video}'")
                test_ids = rng.choice(track_ids, n_test, replace=False)
                test_data.append(tracks[tracks["label"].isin(test_ids)])
                train_data.append(tracks[~tracks["label"].isin(test_ids)])

        train_data = TracksDataFrame(pd.concat(train_data, ignore_index=True))
        test_data = TracksDataFrame(pd.concat(test_data, ignore_index=True))

        return train_data, test_data

    def compute_displacements(self) -> np.ndarray:
        """Compute per-step displacement magnitudes across all trajectories.

        Handles gaps in frame continuity by splitting trajectories at
        missing frames before computing displacements, so step sizes are
        never computed across discontinuities.

        Returns
        -------
        np.ndarray
            1D array of displacement magnitudes, one per consecutive
            detection pair across all trajectories and videos.
        """
        displacements = []

        for video in self["set"].unique():
            video_data = self[self["set"] == video]
            for particle in video_data["label"].unique():
                particle_selection = video_data[video_data["label"] == particle]
                frame_gaps = np.where(np.diff(particle_selection["frame"]) > 1)[0]
                coords = particle_selection[["centroid-0", "centroid-1"]].to_numpy()
                for section in np.split(coords, frame_gaps + 1):
                    step_displacement = np.linalg.norm(np.diff(section, axis=0), axis=1)
                    displacements.append(step_displacement)

        displacements = np.concatenate(displacements)
        return displacements
