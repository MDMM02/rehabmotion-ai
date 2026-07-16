import math

import pytest

from rehabmotion.biomechanics.angles import calculate_angle, calculate_range_of_motion
from rehabmotion.biomechanics.kinematics import calculate_trunk_lean


def test_calculate_angle_for_straight_joint() -> None:
    assert calculate_angle((0, 0), (1, 0), (2, 0)) == pytest.approx(180.0)


def test_calculate_angle_for_right_angle() -> None:
    assert calculate_angle((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)


def test_degenerate_angle_returns_nan() -> None:
    assert math.isnan(calculate_angle((0, 0), (0, 0), (1, 1)))


def test_range_of_motion_ignores_nan() -> None:
    assert calculate_range_of_motion([170.0, float("nan"), 90.0]) == 80.0


def test_trunk_lean_is_measured_from_vertical() -> None:
    assert calculate_trunk_lean((0, 0), (0, 2)) == pytest.approx(0.0)
    assert calculate_trunk_lean((2, 0), (0, 0)) == pytest.approx(90.0)
