import numpy as np
from typing import Optional


def compute_trajectory_features(coords: np.ndarray, frame_rate: Optional[float] = None) -> dict:
    """Compute hand-crafted features from a single particle trajectory.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (T, 2) containing (x, y) coordinates in temporal order.
    frame_rate : float, optional
        Frame rate in Hz, used to convert frame-based features to physical
        units. If None, features are in frame units.

    Returns
    -------
    dict
        Dictionary of scalar features describing the trajectory.
    """
    dt = 1 / frame_rate if frame_rate is not None else 1.0
    T = len(coords)

    # Displacements and speeds
    displacements = np.diff(coords, axis=0)                          # (T-1, 2)
    step_sizes = np.linalg.norm(displacements, axis=1)              # (T-1,)
    speeds = step_sizes / dt

    # --- Diffusion ---
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)

    # MSD at increasing lag times (up to T//4 for statistical reliability)
    max_lag = max(1, T // 4)
    msd = np.array([
        np.mean(np.sum((coords[lag:] - coords[:-lag]) ** 2, axis=1))
        for lag in range(1, max_lag + 1)
    ])
    lags = np.arange(1, max_lag + 1) * dt

    # Anomalous diffusion exponent alpha from log-log fit of MSD vs lag
    # MSD ~ D * t^alpha: alpha=1 Brownian, alpha<1 confined, alpha>2 directed
    if max_lag >= 2:
        log_lags = np.log(lags)
        log_msd = np.log(msd + 1e-10)
        alpha, log_D = np.polyfit(log_lags, log_msd, 1)
        diffusion_coeff = np.exp(log_D) / 4  # 2D: MSD = 4Dt
    else:
        alpha, diffusion_coeff = np.nan, np.nan

    # --- Confinement ---
    center = np.mean(coords, axis=0)
    radius_of_gyration = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
    max_displacement = np.max(np.linalg.norm(coords - coords[0], axis=1))
    net_displacement = np.linalg.norm(coords[-1] - coords[0])

    # Confinement ratio: ratio of net displacement to total path length
    total_path_length = np.sum(step_sizes)
    confinement_ratio = net_displacement / (total_path_length + 1e-10)

    # --- Directionality ---
    # Velocity autocorrelation at lag 1: measures persistence
    if T > 2:
        v = displacements / (step_sizes[:, None] + 1e-10)  # unit vectors
        velocity_autocorr = np.mean(np.sum(v[:-1] * v[1:], axis=1))
    else:
        velocity_autocorr = np.nan

    # Straightness: ratio of end-to-end distance to total path length
    straightness = net_displacement / (total_path_length + 1e-10)

    # Mean turning angle
    if T > 2:
        cos_angles = np.clip(np.sum(v[:-1] * v[1:], axis=1), -1, 1)
        mean_turning_angle = np.mean(np.arccos(cos_angles))
    else:
        mean_turning_angle = np.nan

    # --- Asymmetry ---
    # Gyration tensor eigenvalues — measure of anisotropy
    centered = coords - center
    gyration_tensor = (centered.T @ centered) / T
    eigenvalues = np.linalg.eigvalsh(gyration_tensor)
    asymmetry = 1 - (eigenvalues[0] / (eigenvalues[1] + 1e-10))  # 0=isotropic, 1=linear

    return {
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "alpha": alpha,                          # anomalous diffusion exponent
        "diffusion_coeff": diffusion_coeff,
        "radius_of_gyration": radius_of_gyration,
        "max_displacement": max_displacement,
        "net_displacement": net_displacement,
        "confinement_ratio": confinement_ratio,
        "velocity_autocorr": velocity_autocorr,  # persistence
        "straightness": straightness,
        "mean_turning_angle": mean_turning_angle,
        "asymmetry": asymmetry,
    }