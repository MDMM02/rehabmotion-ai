# Limitations and responsible use

## Regulatory and clinical scope

- RehabMotion AI is an educational and R&D prototype, not a medical device.
- The software has no clinical validation, regulatory clearance or quality-management process.
- Results must not be used for diagnosis, treatment decisions, return-to-activity decisions or patient monitoring.
- Tempo labels and repetition thresholds are descriptive heuristics, not clinical scores.

## Measurement limitations

- A single RGB camera cannot reproduce 3D motion capture.
- Knee and hip values are 2D image-plane angles and depend on camera placement.
- Perspective, camera roll, lens distortion and out-of-plane movement introduce error.
- Trunk lean is unsigned and does not distinguish forward from backward inclination.
- ROM is computed from sampled visible frames and is not equivalent to goniometry.
- There is no anthropometric calibration, force estimation or kinetic analysis.
- Left-right asymmetry requires both sides to be visible in the same frames.

## Pose and video limitations

- Only one person is analyzed.
- Cropping, occlusion, motion blur, poor lighting and loose clothing reduce accuracy.
- External objects can hide landmarks or create implausible skeleton positions.
- Side-view recordings often make the far limb unreliable.
- Long low-confidence gaps remain missing and can prevent repetition detection.
- Processing at a reduced frame rate can miss very fast movements.

## Algorithmic limitations

- Repetition detection assumes a simple, repeated knee-angle cycle.
- Incomplete cycles at the beginning or end of a clip are discarded.
- Shallow or irregular movements may not cross the adaptive thresholds.
- Exercise selection controls signal polarity and must match the recording.
- Savitzky-Golay smoothing can reduce short-duration peaks.
- No manual correction interface is currently provided.
- Metrics do not include uncertainty intervals.

## Data and privacy

- Videos can contain identifiable people and should only be processed with appropriate permission.
- The application uses temporary local files for uploaded videos and removes them after processing.
- Exported CSV/PDF files can still contain sensitive movement data and must be handled appropriately.
- The MediaPipe model runs locally after its first download; the application does not implement cloud storage.

> This prototype is for educational and R&D purposes only. It is not a certified medical device and should not be used for diagnosis, treatment decisions or clinical monitoring.
