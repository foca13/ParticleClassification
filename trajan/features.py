import numpy as np
from typing import Optional


def compute_trajectory_features(coords: np.ndarray, frame_rate: Optional[float] = None, window_size: int = 10) -> dict:
    """Compute hand-crafted features from a single particle trajectory.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (T, 2) containing (x, y) coordinates in temporal order.
    frame_rate : float, optional
        Frame rate in Hz, used to convert frame-based features to physical
        units. If None, features are in frame units.
    window_size : int, optional
        Size of the sliding window in frames used to compute local features
        such as local alpha and pause fraction. Default is 10.

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

    # Confinement index: ratio of MSD at short lags to MSD at long lags
    # Values close to 1 indicate free diffusion, values < 1 indicate confinement
    if max_lag >= 4:
        short_lag_msd = np.mean(msd[:max_lag // 4])
        long_lag_msd = np.mean(msd[3 * max_lag // 4:])
        confinement_index = short_lag_msd / (long_lag_msd + 1e-10)
    else:
        confinement_index = np.nan

    # --- Confinement ---
    center = np.mean(coords, axis=0)
    radius_of_gyration = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
    max_displacement = np.max(np.linalg.norm(coords - coords[0], axis=1))
    net_displacement = np.linalg.norm(coords[-1] - coords[0])

    # Confinement ratio: ratio of net displacement to total path length
    total_path_length = np.sum(step_sizes)
    confinement_ratio = net_displacement / (total_path_length + 1e-10)

    # Pause fraction: fraction of steps where speed is below 10% of mean speed
    # Captures transient confinement and stop-and-go transport
    pause_threshold = 0.1 * mean_speed
    pause_fraction = np.mean(speeds < pause_threshold)

    # --- Directionality ---
    if T > 2:
        v = displacements / (step_sizes[:, None] + 1e-10)  # unit vectors

        # Velocity autocorrelation at lag 1: measures persistence
        velocity_autocorr_lag1 = np.mean(np.sum(v[:-1] * v[1:], axis=1))

        # Velocity autocorrelation at lags 1, 2, 3 — decay rate indicates persistence length
        vacf = np.array([
            np.mean(np.sum(v[:-lag] * v[lag:], axis=1))
            for lag in range(1, min(4, T - 1))
        ])
        vacf_decay = vacf[0] - vacf[-1]  # total decay across measured lags
    else:
        velocity_autocorr_lag1 = np.nan
        vacf_decay = np.nan

    # Straightness: ratio of end-to-end distance to total path length
    straightness = net_displacement / (total_path_length + 1e-10)

    # Mean and std of turning angle
    if T > 2:
        cos_angles = np.clip(np.sum(v[:-1] * v[1:], axis=1), -1, 1)
        turning_angles = np.arccos(cos_angles)
        mean_turning_angle = np.mean(turning_angles)
        std_turning_angle = np.std(turning_angles)
    else:
        mean_turning_angle = np.nan
        std_turning_angle = np.nan

    # Directionality ratio at multiple scales (short, medium, long)
    # Measures net displacement / path length at different time intervals
    if T >= 6:
        short_scale = T // 3
        mid_scale = 2 * T // 3
        dr_short = np.mean([
            np.linalg.norm(coords[i + short_scale] - coords[i]) /
            (np.sum(step_sizes[i:i + short_scale]) + 1e-10)
            for i in range(T - short_scale)
        ])
        dr_mid = np.mean([
            np.linalg.norm(coords[i + mid_scale] - coords[i]) /
            (np.sum(step_sizes[i:i + mid_scale]) + 1e-10)
            for i in range(T - mid_scale)
        ])
        directionality_ratio_short = dr_short
        directionality_ratio_mid = dr_mid
    else:
        directionality_ratio_short = np.nan
        directionality_ratio_mid = np.nan

    # --- Local alpha in sliding windows ---
    # Mean and std of local anomalous diffusion exponent
    # Captures switching between motion modes (directed/diffusive/confined)
    local_alphas = []
    if T >= 2 * window_size:
        for i in range(T - window_size):
            window = coords[i:i + window_size]
            local_max_lag = max(1, window_size // 4)
            if local_max_lag < 2:
                continue
            local_msd = np.array([
                np.mean(np.sum((window[lag:] - window[:-lag]) ** 2, axis=1))
                for lag in range(1, local_max_lag + 1)
            ])
            local_lags = np.arange(1, local_max_lag + 1) * dt
            try:
                local_alpha, _ = np.polyfit(np.log(local_lags), np.log(local_msd + 1e-10), 1)
                local_alphas.append(local_alpha)
            except (np.linalg.LinAlgError, ValueError):
                continue

    if local_alphas:
        mean_local_alpha = np.mean(local_alphas)
        std_local_alpha = np.std(local_alphas)
        range_local_alpha = np.ptp(local_alphas)
    else:
        mean_local_alpha = std_local_alpha = range_local_alpha = np.nan

    # --- Statistical features ---
    # Non-Gaussianity parameter: measures deviation from Brownian diffusion
    # For Gaussian displacements NGP = 0; positive values indicate anomalous diffusion
    if T > 2:
        disp_sq = np.sum(displacements ** 2, axis=1)
        mean_disp_sq = np.mean(disp_sq)
        mean_disp_4th = np.mean(disp_sq ** 2)
        non_gaussianity = mean_disp_4th / (2 * mean_disp_sq ** 2 + 1e-10) - 1
    else:
        non_gaussianity = np.nan

    # Kurtosis of step size distribution
    if T > 4:
        step_mean = np.mean(step_sizes)
        step_std = np.std(step_sizes)
        kurtosis = np.mean(((step_sizes - step_mean) / (step_std + 1e-10)) ** 4) - 3
    else:
        kurtosis = np.nan

    # Entropy of turning angle distribution
    # Uniform distribution (random walk) has high entropy; directed motion has low entropy
    if T > 3:
        hist, _ = np.histogram(turning_angles, bins=8, range=(0, np.pi), density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        turning_angle_entropy = -np.sum(hist * np.log(hist))
    else:
        turning_angle_entropy = np.nan

    # --- Asymmetry ---
    # Gyration tensor eigenvalues — measure of anisotropy
    centered = coords - center
    gyration_tensor = (centered.T @ centered) / T
    eigenvalues = np.linalg.eigvalsh(gyration_tensor)
    asymmetry = 1 - (eigenvalues[0] / (eigenvalues[1] + 1e-10))  # 0=isotropic, 1=linear

    return {
        # Diffusion
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "alpha": alpha,
        "diffusion_coeff": diffusion_coeff,
        "confinement_index": confinement_index,
        # Confinement
        "radius_of_gyration": radius_of_gyration,
        "max_displacement": max_displacement,
        "net_displacement": net_displacement,
        "confinement_ratio": confinement_ratio,
        "pause_fraction": pause_fraction,
        # Directionality
        "velocity_autocorr_lag1": velocity_autocorr_lag1,
        "vacf_decay": vacf_decay,
        "straightness": straightness,
        "mean_turning_angle": mean_turning_angle,
        "std_turning_angle": std_turning_angle,
        "directionality_ratio_short": directionality_ratio_short,
        "directionality_ratio_mid": directionality_ratio_mid,
        # Local alpha
        "mean_local_alpha": mean_local_alpha,
        "std_local_alpha": std_local_alpha,
        "range_local_alpha": range_local_alpha,
        # Statistical
        "non_gaussianity": non_gaussianity,
        "kurtosis": kurtosis,
        "turning_angle_entropy": turning_angle_entropy,
        # Asymmetry
        "asymmetry": asymmetry,
    }
