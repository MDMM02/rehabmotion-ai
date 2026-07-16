# Portfolio demo

## What the demo shows

The GIF and MP4 combine a transformed side-view squat excerpt with:

- MediaPipe pose connections and visibility-colored landmarks;
- the analyzed knee angle on the frame;
- active repetition and movement phase;
- whole-clip knee ROM;
- repetition duration and tempo regularity;
- a moving cursor on the knee-angle signal;
- an explicit R&D / non-medical disclaimer.

The excerpt contains two displayed repetitions from a five-repetition analysis. The raw source video is not stored in the repository.

## Rebuild the assets

First run the application and download the pose-landmark CSV for a video. Then execute:

```bash
python scripts/build_portfolio_assets.py \
  --video path/to/video.mp4 \
  --landmarks path/to/video_pose_landmarks.csv \
  --output-dir docs/assets \
  --exercise squat \
  --side auto \
  --fps 5 \
  --repetitions 2
```

Outputs:

- `docs/assets/rehabmotion_demo.gif` for the README;
- `docs/assets/rehabmotion_demo.mp4` for a higher-quality preview.

## Recorded demonstration metrics

| Rep | Duration | Knee ROM | Hip ROM | Max trunk lean |
|---:|---:|---:|---:|---:|
| 1 | 3.80 s | 96.10 deg | 92.31 deg | 31.00 deg |
| 2 | 3.40 s | 102.12 deg | 100.77 deg | 29.46 deg |
| 3 | 3.30 s | 103.03 deg | 101.87 deg | 30.72 deg |
| 4 | 3.80 s | 105.36 deg | 106.82 deg | 31.55 deg |
| 5 | 3.70 s | 105.19 deg | 108.20 deg | 32.08 deg |

The mean repetition duration is 3.60 seconds and its coefficient of variation is 6.5%, producing the prototype label `high` regularity. These values describe only the demonstration recording.

See [ATTRIBUTION.md](assets/ATTRIBUTION.md) for media provenance and [limitations.md](limitations.md) before interpreting the results.
