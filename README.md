# Eye-Tracking Analysis: Offset Correction & Fixation Detection

A clean, well-documented Python toolkit for processing remote eye-tracking
data from scene-viewing experiments.

Implements the **I-DT (Identification by Dispersion-Threshold)** algorithm,
the gold-standard method for fixation detection (Salvucci & Goldberg, 2000),
together with a static **gaze offset correction** pipeline for multi-site
or remote-tracker datasets.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`01_offset_correction.ipynb`](notebooks/01_offset_correction.ipynb) | Estimates and applies a per-participant static offset correction. Includes visualisation of raw vs corrected gaze. |
| [`02_fixation_analysis.ipynb`](notebooks/02_fixation_analysis.ipynb) | Runs the full I-DT pipeline on corrected gaze data and extracts fixation metrics per trial and AOI. |

---

## Source module

```
src/fixation_detection.py
```

All detection logic lives here. Import it in your own scripts or notebooks:

```python
from src.fixation_detection import FixationDetectionConfig, detect_fixations_complete

points = [{"x": 0.50, "y": 0.40, "t": 0.0},
          {"x": 0.51, "y": 0.41, "t": 16.7},
          ...]                                    # t in milliseconds

config = FixationDetectionConfig(min_duration_ms=100)
fixations = detect_fixations_complete(points, config)

for f in fixations:
    print(f.center_x, f.center_y, f.duration_ms)
```

---

## Data format

Gaze data is represented as a list of dicts:

```python
{
    "x": float,    # normalised horizontal position [0, 1]
    "y": float,    # normalised vertical position   [0, 1]
    "t": float     # timestamp in milliseconds
}
```

The aggregated dataset structure used across notebooks:

```python
aggregated_data = {
    "participant_id": {
        "info":   {"group": "A", ...},
        "trials": {
            1: {"points": [{"x": ..., "y": ..., "t": ...}, ...]},
            2: {"points": [...]},
        }
    }
}
```

---

## Installation

```bash
git clone https://github.com/<your-handle>/eye-tracking-analysis.git
cd eye-tracking-analysis
pip install -r requirements.txt
```

### Requirements

```
numpy
pandas
matplotlib
seaborn
scipy          # only needed for RQA metrics (optional)
jupyter
```

---

## Algorithm overview

### 1. Static offset correction

Remote eye-trackers often produce a systematic spatial offset due to
individual differences in head position and device calibration.

The correction assumes that, over a full session, a participant's average
gaze should coincide with the theoretical centre of the stimulus layout.
The correction vector is:

```
v_i = centre_theoretical − mean_gaze_i
```

All samples are then shifted by `v_i` and clipped to `[0, 1]`.

### 2. I-DT fixation detection

The I-DT algorithm (Salvucci & Goldberg, 2000) classifies consecutive gaze
samples as a fixation when their spatial dispersion stays below a threshold:

```
dispersion = (max_x − min_x) + (max_y − min_y)
```

The full pipeline includes:
1. **Interpolation** of short data gaps (< 75 ms, following I2MC).
2. **Velocity filter** to exclude saccade samples before the I-DT pass.
3. **I-DT** sliding-window algorithm.
4. **Merging** of nearby fixations separated by short gaps (< 75 ms).

### Preset configurations

| Preset | `dispersion_threshold` | `min_duration_ms` | Use case |
|--------|------------------------|-------------------|----------|
| `CONFIG_DEFAULT`      | 0.025 | 100 | Scene viewing, adults |
| `CONFIG_CONSERVATIVE` | 0.020 | 150 | High data quality / strict criteria |
| `CONFIG_PERMISSIVE`   | 0.035 |  80 | Noisy data, children, clinical populations |

---

## References

- Salvucci, D.D., & Goldberg, J.H. (2000). Identifying fixations and saccades in eye-tracking protocols. *ETRA*, 71–78.
- Rayner, K. (2009). Eye movements and attention in reading, scene perception, and visual search. *QJEP*, 62, 1457–1506.
- Nyström, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation, saccade, and glissade detection. *Behavior Research Methods*, 42, 188–204.
- Hessels, R.S., et al. (2017). Noise-robust fixation detection in eye movement data (I2MC). *Behavior Research Methods*, 49, 1693–1713.
- Holmqvist, K., et al. (2011). *Eye Tracking: A Comprehensive Guide to Methods and Measures*. OUP.

---

## Context

Developed as part of a multi-site (8 schools, 500+ participants) eye-tracking
study using progressive matrices stimuli with children, analysing gaze
behaviour differences across socioeconomic groups. Trackers operated at
60–120 Hz with variable per-device calibration quality, motivating the
offset correction step.
