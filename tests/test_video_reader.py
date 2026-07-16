from pathlib import Path

import pytest

from rehabmotion.utils.exceptions import InvalidVideoError
from rehabmotion.video.reader import read_video_metadata


def test_missing_video_raises_clear_error(tmp_path: Path) -> None:
    missing_video = tmp_path / "missing.mp4"

    with pytest.raises(InvalidVideoError, match="does not exist"):
        read_video_metadata(missing_video)


def test_invalid_video_raises_clear_error(tmp_path: Path) -> None:
    invalid_video = tmp_path / "invalid.mp4"
    invalid_video.write_text("not a real video", encoding="utf-8")

    with pytest.raises(InvalidVideoError, match="cannot be read"):
        read_video_metadata(invalid_video)

