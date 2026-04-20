import pathlib
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import torch


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


def merge_tracks(track_recordings: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Flatten a list of recordings into a single list of trajectories.

    Parameters
    ----------
    track_recordings : List[List[np.ndarray]]
        A list of recordings, each being a list of particle trajectories.

    Returns
    -------
    List[np.ndarray]
        A flat list of all trajectories across all recordings.
    """
    return sum(track_recordings, [])


def split_trajectories(
    tracks: List[np.ndarray],
    size: int = 20,
    label: int = 0,
) -> List[Tuple[torch.Tensor, int]]:
    """Split and normalize trajectories into fixed-length segments.

    Each trajectory is divided into non-overlapping segments of exactly
    `size` detections. Segments shorter than `size` are discarded.
    Each segment is origin-centered (first detection subtracted) and
    the temporal axis is normalized by `size`.

    Parameters
    ----------
    tracks : List[np.ndarray]
        A list of trajectories, each an array of shape (T, 3) with
        columns (t, x, y).
    size : int, optional
        The number of detections per segment. Default is 20.
    label : int, optional
        The class label to assign to all segments, used when building
        a labelled dataset from multiple particle types. Default is 0.

    Returns
    -------
    List[Tuple[torch.Tensor, int]]
        A list of (segment, label) pairs. Each segment is a float32
        tensor of shape (3, size) with rows (t, x, y).
    """
    particles = []
    for particle in tracks:
        split_idx = (np.arange(int(len(particle) / size)) + 1) * size
        for traj in np.array_split(particle, split_idx):
            if len(traj) < size:
                continue
            traj -= traj[0]
            traj[:, 0] /= size
            particles.append((torch.tensor(traj, dtype=torch.float32).T, label))
    return particles
