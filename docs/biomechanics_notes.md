# Biomechanics notes

## Phase 3 definitions

- Knee angle: angle hip-knee-ankle, with the knee as the vertex.
- Hip angle: angle shoulder-hip-knee, with the hip as the vertex.
- Trunk lean: unsigned shoulder-to-hip segment inclination from the image vertical.
- Range of motion (ROM): maximum minus minimum of the smoothed valid angle signal.

MediaPipe x and y coordinates are normalized independently by image width and height. Before angle calculation they are converted back to pixel coordinates to preserve the image aspect ratio.

Shoulder, hip, knee and ankle visibility is evaluated per side. Auto mode selects the side with the highest mean minimum visibility. Angles below the chosen visibility threshold are stored as missing values. Only short internal gaps are interpolated; longer low-confidence gaps remain missing.

Signals are smoothed with a Savitzky-Golay filter. Both raw and smoothed values remain available in the kinematics export.

These are 2D single-camera prototype estimates, not validated clinical measurements.

## Phase 4 repetition detection

Repetitions use adaptive hysteresis thresholds derived from the robust low and high percentiles of the smoothed knee-angle signal.

- Squat and knee flexion: extension to flexion to extension.
- Sit-to-stand: seated flexion to standing extension to seated flexion.
- A movement must cross both thresholds and return to its starting posture.
- Shallow partial movements are discarded.
- Segment boundaries are expanded to the nearest local extension/flexion maxima so per-repetition ROM is not limited to the threshold crossings.

Tempo regularity is a descriptive heuristic based on the coefficient of variation of repetition durations: high at or below 10%, moderate at or below 20%, otherwise variable. It is not a clinical score.
