# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the optional sensor-reference trajectory contract."""

import torch

from isaaclab_assets.tasks.sensor_reference import SensorReferenceBuffer, load_sensor_reference


def test_sensor_reference_pt_is_resampled_and_calibrated(tmp_path):
    """Continuous channels align to 50 Hz and encoder metadata is applied once."""
    path = tmp_path / "reference.pt"
    torch.save(
        {
            "t_s": torch.tensor([0.0, 0.01, 0.03]),
            "joint_pos_rad": torch.zeros(3, 8),
            "joint_zero_rad": torch.ones(8) * 0.1,
            "joint_sign": torch.tensor([1, -1] * 4),
            "joint_vel_rad_s": torch.ones(3, 8),
            "gyro_rad_s": torch.zeros(3, 18),
            "contact_lr": torch.tensor([[0.0, 5.0], [5.0, 0.0], [0.0, 0.0]]),
        },
        path,
    )

    reference = load_sensor_reference(path, target_dt_s=0.02)

    assert reference.joint_pos_rad.shape == (2, 8)
    assert torch.allclose(reference.joint_pos_rad[:, 0], torch.tensor([-0.1, -0.1]))
    assert torch.allclose(reference.joint_pos_rad[:, 1], torch.tensor([0.1, 0.1]))
    assert reference.gyro_rad_s.shape == (2, 6, 3)
    assert torch.all((reference.contact_lr >= 0.0) & (reference.contact_lr <= 1.0))


def test_sensor_reference_buffer_assigns_independent_cursors(tmp_path):
    """Each parallel environment can start at a different reference phase."""
    path = tmp_path / "reference.pt"
    torch.save({"joint_pos_rad": torch.arange(32, dtype=torch.float32).reshape(4, 8)}, path)
    reference = load_sensor_reference(path, target_dt_s=0.02)
    buffer = SensorReferenceBuffer(reference, device=torch.device("cpu"), loop=True)
    buffer.reset(slice(None), num_envs=3)

    assert buffer.indices.shape == (3,)
    assert torch.all((buffer.indices >= 0) & (buffer.indices < reference.length))
    before = buffer.indices.clone()
    buffer.advance(1)
    assert torch.equal(buffer.indices, (before + 1) % reference.length)
    assert buffer.sample()["joint_pos_rad"].shape == (3, 8)
