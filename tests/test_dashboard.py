import pandas as pd

from app.components.upload import _analysis_signature, _exercise_key
from rehabmotion.export.csv_exporter import dataframe_to_csv_bytes


def test_analysis_signature_tracks_pose_processing_settings() -> None:
    video = b"example-video"

    base = _analysis_signature(video, min_visibility=0.6, target_fps=10.0)

    assert base == _analysis_signature(video, 0.6, 10.0)
    assert base != _analysis_signature(video, 0.7, 10.0)
    assert base != _analysis_signature(video, 0.6, 15.0)


def test_exercise_label_maps_to_detection_key() -> None:
    assert _exercise_key("Sit-to-stand") == "sit-to-stand"
    assert _exercise_key("Knee flexion") == "knee-flexion"


def test_csv_download_payload_excludes_dataframe_index() -> None:
    payload = dataframe_to_csv_bytes(pd.DataFrame({"angle": [90.0]}))

    assert payload == b"angle\n90.0\n"
