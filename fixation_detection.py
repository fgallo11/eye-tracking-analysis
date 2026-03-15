"""
fixation_detection.py
=====================
Gold-standard fixation detection for remote eye-trackers.

Implements the I-DT (Identification by Dispersion-Threshold) algorithm
described in Salvucci & Goldberg (2000), with:
  - Optional velocity-based saccade filter
  - Linear interpolation of short missing-data gaps (I2MC-style)
  - Post-hoc merging of nearby fixations (Nyström & Holmqvist, 2010)

Designed for scene-viewing tasks with remote eye-trackers (60–120 Hz).
Works with any gaze data in normalised [0, 1] coordinates.

Usage
-----
    from src.fixation_detection import FixationDetectionConfig, detect_fixations_complete

    points = [{"x": 0.5, "y": 0.4, "t": 0.0}, ...]   # t in milliseconds
    config = FixationDetectionConfig()                  # sensible defaults
    fixations = detect_fixations_complete(points, config)

    for fix in fixations:
        print(fix.center_x, fix.center_y, fix.duration_ms)

Data format
-----------
Each gaze sample is a dict with at minimum:
    {
        "x": float,   # normalised horizontal position  [0, 1]
        "y": float,   # normalised vertical position    [0, 1]
        "t": float    # timestamp in milliseconds
    }

References
----------
Salvucci, D.D., & Goldberg, J.H. (2000). Identifying fixations and saccades
    in eye-tracking protocols. ETRA, 71–78.
Rayner, K. (2009). Eye movements and attention in reading, scene perception,
    and visual search. Quarterly Journal of Experimental Psychology, 62, 1457–1506.
Nyström, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation,
    saccade, and glissade detection in eyetracking data. Behavior Research
    Methods, 42, 188–204.
Hessels, R.S., Niehorster, D.C., Kemner, C., & Hooge, I.T. (2017).
    Noise-robust fixation detection in eye movement data. Behavior Research
    Methods, 49, 1693–1713.  [I2MC]
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FixationDetectionConfig:
    """
    Parameters for the I-DT fixation detection algorithm.

    All spatial values are in normalised screen coordinates [0, 1].
    At a typical viewing distance, 1° of visual angle ≈ 0.017–0.030
    normalised units (depends on screen size and distance).

    Attributes
    ----------
    dispersion_threshold : float
        Maximum bounding-box dispersion to classify a window as a fixation.
        Dispersion = (max_x − min_x) + (max_y − min_y).
        Default 0.025 ≈ 1.25° for a standard monitor setup.
        Increase for noisy data or young children; decrease for high-quality
        adult data.

    min_duration_ms : int
        Minimum fixation duration in milliseconds.
        • Reading tasks:         50–100 ms
        • Scene viewing (adult): 100–150 ms   ← default
        • Children / clinical:   150–200 ms

    min_samples : int
        Minimum number of gaze samples to form a fixation.
        Should be consistent with min_duration_ms and the tracker's
        sampling rate (e.g. 6 samples ≈ 100 ms at 60 Hz).

    velocity_threshold : float
        Gaze velocity above which a sample is classified as a saccade
        and excluded before the I-DT pass.
        Units: normalised coordinates per millisecond.
        Default 0.0008 ≈ 40 °/s for a 60 Hz tracker.
        Set use_velocity_filter=False to skip this step.

    use_velocity_filter : bool
        Whether to run the velocity-based pre-filter. Recommended for
        60 Hz data; may be turned off for very-high-frequency trackers
        that already provide clean data.

    merge_threshold_ms : float
        Maximum gap between two consecutive fixations for them to be
        candidates for merging. Default 75 ms (Nyström & Holmqvist, 2010).

    merge_dispersion_factor : float
        Two candidate fixations are merged only if their spatial distance
        is < dispersion_threshold × merge_dispersion_factor.

    max_interpolation_gap_ms : float
        Gaps in the gaze stream shorter than this value are filled with
        linear interpolation before detection (I2MC recommendation: < 100 ms).
        Set to 0 to disable interpolation.
    """

    # --- Core I-DT parameters ---
    dispersion_threshold: float = 0.025
    min_duration_ms: int = 100
    min_samples: int = 6

    # --- Velocity pre-filter ---
    velocity_threshold: float = 0.0008
    use_velocity_filter: bool = True

    # --- Post-processing ---
    merge_threshold_ms: float = 75.0
    merge_dispersion_factor: float = 1.5

    # --- Missing-data interpolation ---
    max_interpolation_gap_ms: float = 75.0

    def __post_init__(self) -> None:
        if not (0 < self.dispersion_threshold < 0.5):
            raise ValueError(
                f"dispersion_threshold must be in (0, 0.5); got {self.dispersion_threshold}"
            )
        if not (20 <= self.min_duration_ms <= 1000):
            raise ValueError(
                f"min_duration_ms must be in [20, 1000]; got {self.min_duration_ms}"
            )
        if self.min_samples < 2:
            raise ValueError(
                f"min_samples must be >= 2; got {self.min_samples}"
            )


# Preset configurations for common use cases
CONFIG_DEFAULT = FixationDetectionConfig()

CONFIG_CONSERVATIVE = FixationDetectionConfig(
    dispersion_threshold=0.020,
    min_duration_ms=150,
    min_samples=9,
    velocity_threshold=0.0006,
    merge_threshold_ms=50.0,
    merge_dispersion_factor=1.2,
)

CONFIG_PERMISSIVE = FixationDetectionConfig(
    dispersion_threshold=0.035,
    min_duration_ms=80,
    min_samples=5,
    velocity_threshold=0.001,
    use_velocity_filter=False,
    merge_threshold_ms=100.0,
    merge_dispersion_factor=2.0,
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Fixation:
    """
    A single detected fixation.

    Attributes
    ----------
    start_time_ms : float   Onset timestamp (ms).
    end_time_ms   : float   Offset timestamp (ms).
    duration_ms   : float   Duration = end − start (ms).
    center_x      : float   Mean horizontal position (normalised).
    center_y      : float   Mean vertical position (normalised).
    dispersion    : float   Bounding-box dispersion of the contributing samples.
    n_samples     : int     Number of gaze samples in this fixation.
    """

    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    center_x: float
    center_y: float
    dispersion: float
    n_samples: int

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)

    def to_dict(self) -> Dict:
        """Serialise to a plain dictionary (useful for DataFrame construction)."""
        return {
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "duration_ms": self.duration_ms,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "dispersion": self.dispersion,
            "n_samples": self.n_samples,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dispersion(points: List[Dict]) -> float:
    """
    Bounding-box dispersion: (max_x − min_x) + (max_y − min_y).

    This is the canonical dispersion metric from Salvucci & Goldberg (2000).
    It is computationally cheaper and more robust to single outliers than
    the maximum pairwise Euclidean distance.
    """
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def _velocity(p1: Dict, p2: Dict) -> float:
    """
    Euclidean velocity between two consecutive samples (normalised units / ms).
    Returns 0 if Δt ≤ 0.
    """
    dt = p2["t"] - p1["t"]
    if dt <= 0:
        return 0.0
    return np.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"]) / dt


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def interpolate_gaps(
    points: List[Dict],
    max_gap_ms: float,
    sample_interval_ms: float = 16.67,
) -> List[Dict]:
    """
    Linearly interpolate missing-data gaps shorter than *max_gap_ms*.

    Gaps are detected as inter-sample intervals that exceed ~1.5× the
    expected sample interval but are shorter than max_gap_ms.  Interpolated
    points are flagged with ``"interpolated": True``.

    Parameters
    ----------
    points            : list of gaze-sample dicts (must have 'x', 'y', 't').
    max_gap_ms        : maximum gap to fill (ms).  Gaps ≥ this are left as-is.
    sample_interval_ms: expected inter-sample interval (ms).
                        Default 16.67 ms = 60 Hz.

    Returns
    -------
    List of gaze samples with short gaps filled.
    """
    if len(points) < 2 or max_gap_ms <= 0:
        return points

    min_gap = 1.5 * sample_interval_ms      # anything above this is a gap
    out = [points[0]]

    for i in range(1, len(points)):
        gap = points[i]["t"] - points[i - 1]["t"]
        if min_gap < gap < max_gap_ms:
            n = max(1, round(gap / sample_interval_ms)) - 1
            for k in range(1, n + 1):
                alpha = k / (n + 1)
                out.append({
                    "x": points[i - 1]["x"] + alpha * (points[i]["x"] - points[i - 1]["x"]),
                    "y": points[i - 1]["y"] + alpha * (points[i]["y"] - points[i - 1]["y"]),
                    "t": points[i - 1]["t"] + alpha * gap,
                    "interpolated": True,
                })
        out.append(points[i])

    return out


def filter_saccades(
    points: List[Dict],
    velocity_threshold: float,
) -> List[Dict]:
    """
    Remove samples whose instantaneous velocity exceeds *velocity_threshold*.

    The first sample is always kept (no preceding sample to compare against).

    Parameters
    ----------
    points             : list of gaze-sample dicts.
    velocity_threshold : velocity cutoff (normalised units / ms).

    Returns
    -------
    Filtered list of gaze samples.
    """
    if not points:
        return points
    filtered = [points[0]]
    for i in range(1, len(points)):
        if _velocity(points[i - 1], points[i]) < velocity_threshold:
            filtered.append(points[i])
    return filtered


def detect_fixations_idt(
    points: List[Dict],
    config: FixationDetectionConfig,
) -> List[Fixation]:
    """
    Core I-DT algorithm (Salvucci & Goldberg, 2000).

    Slides a window of *min_samples* points across the gaze stream.  While
    the bounding-box dispersion of the window stays below
    *dispersion_threshold*, the window is extended by one sample.  When the
    dispersion exceeds the threshold (or data runs out), the current window
    is saved as a fixation if its duration ≥ *min_duration_ms*.

    Parameters
    ----------
    points : list of gaze-sample dicts with 'x', 'y', 't'.
    config : FixationDetectionConfig instance.

    Returns
    -------
    List of Fixation objects, ordered by onset time.
    """
    fixations: List[Fixation] = []
    n = len(points)
    i = 0

    while i <= n - config.min_samples:
        window_end = i + config.min_samples
        window = points[i:window_end]

        if _dispersion(window) > config.dispersion_threshold:
            i += 1
            continue

        # Extend window while dispersion stays below threshold
        while window_end < n:
            candidate = points[i:window_end + 1]
            if _dispersion(candidate) <= config.dispersion_threshold:
                window_end += 1
            else:
                break

        window = points[i:window_end]
        duration = window[-1]["t"] - window[0]["t"]

        if duration >= config.min_duration_ms:
            xs = [p["x"] for p in window]
            ys = [p["y"] for p in window]
            fixations.append(Fixation(
                start_time_ms=window[0]["t"],
                end_time_ms=window[-1]["t"],
                duration_ms=duration,
                center_x=float(np.mean(xs)),
                center_y=float(np.mean(ys)),
                dispersion=_dispersion(window),
                n_samples=len(window),
            ))

        i = window_end  # advance past this fixation

    return fixations


def merge_nearby_fixations(
    fixations: List[Fixation],
    config: FixationDetectionConfig,
) -> List[Fixation]:
    """
    Merge consecutive fixations that are close in both time and space.

    Two fixations are merged when:
      (1) The gap between them is < merge_threshold_ms, AND
      (2) The Euclidean distance between their centres is
          < dispersion_threshold × merge_dispersion_factor.

    The merged fixation's centre is the duration-weighted mean of the two
    original centres.

    Reference: Nyström & Holmqvist (2010), Behavior Research Methods 42, 188–204.

    Parameters
    ----------
    fixations : list of Fixation objects (must be chronologically ordered).
    config    : FixationDetectionConfig instance.

    Returns
    -------
    List of (possibly fewer) Fixation objects after merging.
    """
    if len(fixations) < 2:
        return fixations

    spatial_limit = config.dispersion_threshold * config.merge_dispersion_factor
    merged: List[Fixation] = []
    current = fixations[0]

    for nxt in fixations[1:]:
        time_gap = nxt.start_time_ms - current.end_time_ms
        spatial_gap = np.hypot(
            nxt.center_x - current.center_x,
            nxt.center_y - current.center_y,
        )

        if time_gap <= config.merge_threshold_ms and spatial_gap <= spatial_limit:
            total = current.duration_ms + nxt.duration_ms
            cx = (current.center_x * current.duration_ms + nxt.center_x * nxt.duration_ms) / total
            cy = (current.center_y * current.duration_ms + nxt.center_y * nxt.duration_ms) / total
            current = Fixation(
                start_time_ms=current.start_time_ms,
                end_time_ms=nxt.end_time_ms,
                duration_ms=nxt.end_time_ms - current.start_time_ms,
                center_x=cx,
                center_y=cy,
                dispersion=max(current.dispersion, nxt.dispersion, spatial_gap),
                n_samples=current.n_samples + nxt.n_samples,
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_fixations_complete(
    points: List[Dict],
    config: Optional[FixationDetectionConfig] = None,
    interpolate: bool = True,
    merge: bool = True,
    sample_interval_ms: float = 16.67,
) -> List[Fixation]:
    """
    Full fixation detection pipeline.

    Steps
    -----
    1. (Optional) Interpolate short missing-data gaps.
    2. (Optional) Remove saccade samples by velocity threshold.
    3. Run the I-DT algorithm.
    4. (Optional) Merge nearby fixations.

    Parameters
    ----------
    points             : list of dicts with 'x', 'y', 't'.
                         x and y must be normalised to [0, 1].
                         t must be in milliseconds.
    config             : FixationDetectionConfig.  Defaults to CONFIG_DEFAULT.
    interpolate        : If True, fill short data gaps before detection.
    merge              : If True, merge temporally/spatially close fixations.
    sample_interval_ms : Expected inter-sample interval (ms).
                         Used only when interpolate=True.

    Returns
    -------
    List of Fixation objects, ordered by onset time.
    Empty list if fewer than min_samples points are provided.

    Examples
    --------
    >>> points = [{"x": 0.50, "y": 0.40, "t": 0.0},
    ...           {"x": 0.51, "y": 0.41, "t": 16.7},
    ...           {"x": 0.50, "y": 0.40, "t": 33.4},
    ...           {"x": 0.51, "y": 0.39, "t": 50.1},
    ...           {"x": 0.50, "y": 0.40, "t": 66.8},
    ...           {"x": 0.50, "y": 0.40, "t": 83.5},
    ...           {"x": 0.50, "y": 0.41, "t": 100.2}]
    >>> config = FixationDetectionConfig(min_duration_ms=80)
    >>> fixations = detect_fixations_complete(points, config)
    >>> len(fixations)
    1
    >>> round(fixations[0].center_x, 2)
    0.5
    """
    if config is None:
        config = CONFIG_DEFAULT

    if len(points) < config.min_samples:
        return []

    # Step 1: interpolation
    if interpolate and config.max_interpolation_gap_ms > 0:
        points = interpolate_gaps(points, config.max_interpolation_gap_ms, sample_interval_ms)

    # Step 2: velocity filter
    if config.use_velocity_filter:
        points = filter_saccades(points, config.velocity_threshold)
        if len(points) < config.min_samples:
            return []

    # Step 3: I-DT
    fixations = detect_fixations_idt(points, config)

    # Step 4: merge
    if merge and len(fixations) > 1:
        fixations = merge_nearby_fixations(fixations, config)

    return fixations


def fixations_to_records(fixations: List[Fixation]) -> List[Dict]:
    """
    Convert a list of Fixation objects to a list of plain dicts.

    Useful for constructing a pandas DataFrame::

        import pandas as pd
        df = pd.DataFrame(fixations_to_records(fixations))
    """
    return [f.to_dict() for f in fixations]
