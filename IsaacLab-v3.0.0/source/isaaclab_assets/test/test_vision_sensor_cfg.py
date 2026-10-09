# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests for the fixed camera placement contract."""

import importlib.util
import math

import pytest
import torch

from isaaclab.utils.math import matrix_from_quat

from isaaclab_assets.tasks.vision_sensor_cfg import (
    CAMERA_FOCAL_LENGTH_CM,
    CHANNEL_LENGTH_M,
    DEFAULT_CHANNEL_HALF_WIDTH_M,
    EVENT_CAMERA_NAMES,
    RGB_CAMERA_NAMES,
    fixed_camera_pose,
    horizontal_fov_deg,
    look_at_quaternion,
    make_event_camera_cfg,
)


def test_fixed_camera_poses_are_mirrored_and_look_at_is_normalized():
    """Both side rigs target the same channel centre with unit quaternions."""
    left_pos, left_rot = fixed_camera_pose("left")
    right_pos, right_rot = fixed_camera_pose("right")

    assert left_pos == (1.2, 1.2, 1.0)
    assert right_pos == (1.2, -1.2, 1.0)
    assert sum(value * value for value in left_rot) == pytest.approx(1.0, abs=1e-5)
    assert sum(value * value for value in right_rot) == pytest.approx(1.0, abs=1e-5)


def test_look_at_rejects_degenerate_pose():
    """A camera cannot be oriented when its target equals its position."""
    with pytest.raises(ValueError, match="distinct"):
        look_at_quaternion((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@pytest.mark.parametrize("side", ["left", "right"])
def test_look_at_forward_axis_points_at_channel_center(side):
    position, rotation = fixed_camera_pose(side)
    target = (1.2, 0.0, 0.70)
    matrix = matrix_from_quat(torch.tensor(rotation).reshape(1, 4))[0]
    camera_forward = matrix @ torch.tensor((0.0, 0.0, -1.0))
    expected = torch.tensor(target) - torch.tensor(position)
    expected = expected / torch.linalg.norm(expected)
    assert torch.allclose(camera_forward, expected, atol=1.0e-5)


def test_camera_fov_covers_both_ends_of_the_channel():
    required_half_angle = math.degrees(
        math.atan2(CHANNEL_LENGTH_M / 2.0, DEFAULT_CHANNEL_HALF_WIDTH_M)
    )
    assert horizontal_fov_deg() > 2.0 * required_half_angle + 2.0
    assert CAMERA_FOCAL_LENGTH_CM < 18.0


def test_event_cfg_never_silently_falls_back_to_rgb_difference():
    """Stock Isaac Lab has no event sensor and must fail explicitly."""
    if importlib.util.find_spec("dvs_gen") is not None:
        pytest.skip("EVIS is installed; this test covers the stock Isaac Lab path")
    with pytest.raises(RuntimeError, match="dvs_gen"):
        make_event_camera_cfg("left_event_camera", "left")


def test_camera_name_contract_is_stable():
    """Keep the names aligned with task mapping and dataset conventions."""
    assert RGB_CAMERA_NAMES == ("left_rgb_camera", "right_rgb_camera")
    assert EVENT_CAMERA_NAMES == ("left_event_camera", "right_event_camera")


def test_rgb_and_deferred_event_cameras_share_wide_lens_calibration():
    from isaaclab_assets.envs.my_exoskeleton_imitation_env_cfg import MyExoskeletonImitationEnvCfg

    cfg = MyExoskeletonImitationEnvCfg()
    assert cfg.scene.left_rgb_camera.spawn.focal_length == CAMERA_FOCAL_LENGTH_CM
    assert cfg.scene.right_rgb_camera.spawn.focal_length == CAMERA_FOCAL_LENGTH_CM
    assert cfg.scene.left_event_camera.spawn.focal_length == CAMERA_FOCAL_LENGTH_CM
    assert cfg.scene.right_event_camera.spawn.focal_length == CAMERA_FOCAL_LENGTH_CM
