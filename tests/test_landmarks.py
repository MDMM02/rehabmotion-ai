from types import SimpleNamespace

from rehabmotion.pose.landmarks import records_from_landmarks, records_to_dataframe


def test_landmarks_are_exported_in_long_format() -> None:
    landmarks = [
        SimpleNamespace(
            x=index / 100,
            y=index / 50,
            z=-index / 1000,
            visibility=0.9,
            presence=0.95,
        )
        for index in range(33)
    ]

    records = records_from_landmarks(landmarks, frame=12, timestamp_seconds=0.4)
    dataframe = records_to_dataframe(records)

    assert len(records) == 33
    assert records[25].landmark_name == "left_knee"
    assert dataframe.loc[26, "landmark_name"] == "right_knee"
    assert dataframe.loc[0, "frame"] == 12
