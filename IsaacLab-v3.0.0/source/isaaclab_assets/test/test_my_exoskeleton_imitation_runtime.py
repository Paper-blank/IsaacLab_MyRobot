"""Dependency-light shape checks for the EVIS event bridge."""

from __future__ import annotations

import pytest
import torch

from isaaclab_assets.tasks.imitation_reward import (
    camera_rgb_observation,
    event_camera_observation,
    imitation_event_reward,
    imitation_imu_reward,
    imu_observation,
)
from isaaclab_assets.tasks.my_exoskeleton_flat import (
    _foot_contacts,
    _foot_pressure_features,
    short_contact_reward,
)
from isaaclab_assets.tasks.my_exoskeleton_imitation import _event_mask_batch, build_imitation_diagnostics


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((8, 10), (4, 8, 10)),
        ((4, 8, 10), (4, 8, 10)),
        ((1, 8, 10), (4, 8, 10)),
        ((4, 1, 8, 10), (4, 8, 10)),
        ((1, 4, 8, 10), (4, 8, 10)),
    ],
)
def test_event_mask_batch_normalizes_single_and_batched_masks(shape, expected):
    value = _event_mask_batch(torch.ones(shape), 4, "cpu")
    assert value is not None
    assert tuple(value.shape) == expected


def test_event_mask_batch_rejects_ambiguous_environment_axis():
    assert _event_mask_batch(torch.ones((2, 8, 10)), 4, "cpu") is None
    assert _event_mask_batch(torch.ones((2, 3, 8, 10)), 4, "cpu") is None


def test_diagnostics_snapshot_is_compact_and_json_safe():
    class Sim:
        current_time = 1.25

    class RewardManager:
        _reward_buf = torch.tensor([0.5, 1.5])

    class Env:
        num_envs = 2
        _diagnostics_step = 20
        sim = Sim()
        reward_manager = RewardManager()
        _latest_imu = torch.ones((2, 6, 9))
        _latest_imu_valid = torch.ones((2, 6), dtype=torch.bool)
        _latest_rgb_feature = torch.zeros((2, 12))
        _latest_camera_rgb = {"left_rgb_camera": torch.zeros((2, 256, 256, 3))}
        _latest_event_hist = torch.zeros((2, 32))
        _latest_event_valid = torch.tensor([True, False])
        _imitation_task_ids_per_env = ["BT02", "BT04"]

    snapshot = build_imitation_diagnostics(Env())
    assert snapshot["step"] == 20
    assert snapshot["num_envs"] == 2
    assert snapshot["tasks"] == {"BT02": 1, "BT04": 1}
    assert snapshot["imu"]["shape"] == [2, 6, 9]
    assert snapshot["camera_rgb_frames"]["left_rgb_camera"]["shape"] == [2, 256, 256, 3]
    assert snapshot["event_valid_fraction"] == pytest.approx(0.5)
    assert snapshot["imitation_reward"]["mean"] == pytest.approx(1.0)


def test_event_reward_masks_decoded_target_when_runtime_evis_is_unavailable():
    class Env:
        num_envs = 2
        device = torch.device("cpu")
        _imitation_targets = {
            "target_event_feature": torch.ones((2, 32)),
            "event_valid": torch.ones(2, dtype=torch.bool),
        }

        @staticmethod
        def update_event_observation():
            return torch.zeros((2, 32)), torch.zeros(2, dtype=torch.bool)

    reward = imitation_event_reward(Env())
    assert torch.equal(reward, torch.zeros(2))


def test_disabled_sensor_observations_are_fixed_shape_zero_and_invalid():
    """Ablating optional channels must not make manager shape probes fail."""

    class Env:
        num_envs = 3
        device = torch.device("cpu")
        cfg = type("Cfg", (), {"use_imu": False, "use_rgb": False, "use_event": False})()

    env = Env()
    imu = imu_observation(env)
    rgb = camera_rgb_observation(env)
    event = event_camera_observation(env)
    assert tuple(imu.shape) == (3, 54)
    assert tuple(rgb.shape) == (3, 12)
    assert tuple(event.shape) == (3, 32)
    assert torch.count_nonzero(imu) == 0
    assert torch.count_nonzero(rgb) == 0
    assert torch.count_nonzero(event) == 0
    assert not bool(env._latest_imu_valid.any())
    assert not bool(env._latest_rgb_valid.any())
    assert not bool(env._latest_event_valid.any())
    assert torch.equal(imitation_imu_reward(env), torch.zeros(3))
    assert torch.equal(imitation_event_reward(env), torch.zeros(3))


def test_disabled_contact_sensor_never_reads_scene_entities():
    """Contact ablation returns zeros even when the scene has no contact keys."""

    class Env:
        num_envs = 2
        device = torch.device("cpu")
        cfg = type("Cfg", (), {"use_contact": False})()
        scene = {}

    env = Env()
    left, right = _foot_contacts(env)
    assert torch.equal(left, torch.zeros(2))
    assert torch.equal(right, torch.zeros(2))
    assert torch.equal(_foot_pressure_features(env), torch.zeros((2, 4)))
    assert torch.equal(short_contact_reward(env), torch.zeros(2))
