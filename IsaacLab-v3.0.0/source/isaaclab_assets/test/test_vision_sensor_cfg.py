# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""固定双侧 RGB/事件相机几何配置的无 Kit 回归测试。

本文件保护左右相机镜像位置、朝向通道中心的四元数、覆盖2.4 m通道的视场角、稳定的
传感器名称和共享镜头参数。事件相机插件缺失时必须明确报错或使用标记无效的适配器，
绝不能悄悄改成 RGB 帧差并让训练任务含义发生变化。
"""

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
    """左右相机位置应关于通道镜像，并用单位四元数共同朝向通道中心。"""
    left_pos, left_rot = fixed_camera_pose("left")
    right_pos, right_rot = fixed_camera_pose("right")

    assert left_pos == (1.2, 1.2, 1.0)
    assert right_pos == (1.2, -1.2, 1.0)
    assert sum(value * value for value in left_rot) == pytest.approx(1.0, abs=1e-5)
    assert sum(value * value for value in right_rot) == pytest.approx(1.0, abs=1e-5)


def test_look_at_rejects_degenerate_pose():
    """相机目标点与相机位置重合时无法确定朝向，必须拒绝该输入。"""
    with pytest.raises(ValueError, match="distinct"):
        look_at_quaternion((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@pytest.mark.parametrize("side", ["left", "right"])
def test_look_at_forward_axis_points_at_channel_center(side):
    """把相机局部 ``-Z`` 前向轴旋转到世界坐标后，它应精确指向通道中心。"""

    position, rotation = fixed_camera_pose(side)
    target = (1.2, 0.0, 0.70)
    matrix = matrix_from_quat(torch.tensor(rotation).reshape(1, 4))[0]
    camera_forward = matrix @ torch.tensor((0.0, 0.0, -1.0))
    expected = torch.tensor(target) - torch.tensor(position)
    expected = expected / torch.linalg.norm(expected)
    assert torch.allclose(camera_forward, expected, atol=1.0e-5)


def test_camera_fov_covers_both_ends_of_the_channel():
    """水平视场必须比刚好看见2.4 m通道两端再多留2度余量。"""

    required_half_angle = math.degrees(math.atan2(CHANNEL_LENGTH_M / 2.0, DEFAULT_CHANNEL_HALF_WIDTH_M))
    assert horizontal_fov_deg() > 2.0 * required_half_angle + 2.0
    assert CAMERA_FOCAL_LENGTH_CM < 18.0


def test_event_cfg_never_silently_falls_back_to_rgb_difference():
    """未安装 EVIS 时必须明确报错，不能把 RGB 帧差伪装成真实事件相机。"""
    if importlib.util.find_spec("dvs_gen") is not None:
        pytest.skip("EVIS is installed; this test covers the stock Isaac Lab path")
    with pytest.raises(RuntimeError, match="dvs_gen"):
        make_event_camera_cfg("left_event_camera", "left")


def test_camera_name_contract_is_stable():
    """固定名称必须与任务映射和数据集字段一致，否则运行时会找不到对应相机。"""
    assert RGB_CAMERA_NAMES == ("left_rgb_camera", "right_rgb_camera")
    assert EVENT_CAMERA_NAMES == ("left_event_camera", "right_event_camera")


def test_rgb_and_deferred_event_cameras_share_wide_lens_calibration():
    """RGB 与延迟创建的事件相机必须共用同一广角焦距，保证两类图像空间对齐。"""

    from isaaclab_assets.envs.my_exoskeleton_imitation_env_cfg import MyExoskeletonImitationEnvCfg

    cfg = MyExoskeletonImitationEnvCfg()
    assert cfg.scene.left_rgb_camera.spawn.focal_length == CAMERA_FOCAL_LENGTH_CM
    assert cfg.scene.right_rgb_camera.spawn.focal_length == CAMERA_FOCAL_LENGTH_CM
    assert cfg.scene.left_event_camera.spawn.focal_length == CAMERA_FOCAL_LENGTH_CM
    assert cfg.scene.right_event_camera.spawn.focal_length == CAMERA_FOCAL_LENGTH_CM
