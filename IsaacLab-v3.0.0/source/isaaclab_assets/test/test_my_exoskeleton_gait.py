# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the my_exoskeleton gait and curriculum contract."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab_assets.tasks import gait, my_exoskeleton_flat


def test_gait_modes_have_stable_contiguous_policy_indices():
    """The gait one-hot observation relies on this exact five-channel order."""
    assert list(gait.GaitMode) == [
        gait.GaitMode.STAND,
        gait.GaitMode.WALK,
        gait.GaitMode.WALK_TO_STAND,
        gait.GaitMode.RUN,
        gait.GaitMode.RUN_TO_WALK,
    ]
    assert [mode.value for mode in gait.GaitMode] == list(range(gait.GAIT_MODE_COUNT))
    assert gait.GAIT_MODE_COUNT == 5


def test_froude_number_reaches_the_running_threshold_at_critical_speed():
    """The run boundary must follow Fr = v^2 / (g l), including its units."""
    critical_speed = math.sqrt(gait.FROUDE_THRESHOLD * gait.GRAVITY_M_S2 * gait.LEG_LENGTH_M)

    assert gait.froude_number(0.0) == 0.0
    assert gait.froude_number(critical_speed) == pytest.approx(gait.FROUDE_THRESHOLD)
    assert gait.froude_number(-critical_speed) == pytest.approx(gait.FROUDE_THRESHOLD)


@pytest.mark.parametrize("leg_length_m", [0.0, -0.1])
def test_froude_number_rejects_non_positive_leg_length(leg_length_m):
    with pytest.raises(ValueError, match="leg_length_m must be positive"):
        gait.froude_number(1.0, leg_length_m=leg_length_m)


def test_curriculum_phase_switches_at_inclusive_boundaries():
    phase2_start = 100
    phase3_start = 250

    assert gait.curriculum_phase_for_step(0, phase2_start, phase3_start) is gait.CurriculumPhase.WALK_ONLY
    assert (
        gait.curriculum_phase_for_step(phase2_start - 1, phase2_start, phase3_start) is gait.CurriculumPhase.WALK_ONLY
    )
    assert gait.curriculum_phase_for_step(phase2_start, phase2_start, phase3_start) is gait.CurriculumPhase.TRANSITIONS
    assert (
        gait.curriculum_phase_for_step(phase3_start - 1, phase2_start, phase3_start) is gait.CurriculumPhase.TRANSITIONS
    )
    assert gait.curriculum_phase_for_step(phase3_start, phase2_start, phase3_start) is gait.CurriculumPhase.MULTI_GAIT


@pytest.mark.parametrize("phase2_start, phase3_start", [(-1, 10), (11, 10)])
def test_curriculum_phase_rejects_invalid_boundaries(phase2_start, phase3_start):
    with pytest.raises(ValueError, match="non-negative and ordered"):
        gait.curriculum_phase_for_step(0, phase2_start, phase3_start)


def test_gait_mask_routes_each_environment_to_exactly_one_mode():
    """All mode-specific reward groups must be mutually exclusive per environment."""
    gait_modes = torch.tensor([mode.value for mode in gait.GaitMode], dtype=torch.long)

    class FakeCommandTerm:
        gait_mode = gait_modes

    class FakeCommandManager:
        @staticmethod
        def get_term(name):
            assert name == "base_velocity"
            return FakeCommandTerm()

    class FakeEnv:
        command_manager = FakeCommandManager()

    masks = torch.stack([gait.gait_mask(FakeEnv(), mode) for mode in gait.GaitMode], dim=-1)

    assert masks.dtype == torch.float32
    assert torch.equal(masks, torch.eye(gait.GAIT_MODE_COUNT))
    assert torch.equal(masks.sum(dim=-1), torch.ones(gait.GAIT_MODE_COUNT))
    assert torch.equal(gait.gait_mask(FakeEnv(), gait.GaitMode.RUN.value), masks[:, gait.GaitMode.RUN])


def test_gait_mask_rejects_a_command_without_gait_state():
    class FakeCommandManager:
        @staticmethod
        def get_term(_name):
            return object()

    class FakeEnv:
        command_manager = FakeCommandManager()

    with pytest.raises(TypeError, match="base_velocity must expose a gait_mode tensor"):
        gait.gait_mask(FakeEnv(), gait.GaitMode.WALK)


def test_mode_specific_reward_is_zero_outside_its_route():
    """A representative routed reward must not leak into other gait objectives."""
    gait_modes = torch.tensor([gait.GaitMode.WALK, gait.GaitMode.RUN, gait.GaitMode.RUN_TO_WALK], dtype=torch.long)

    class FakeData:
        root_lin_vel_b = torch.tensor([[2.0, 0.0, 0.0]]).repeat(3, 1)
        root_ang_vel_b = torch.zeros(3, 3)
        root_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)

    class FakeRobot:
        data = FakeData()

    class FakeCommandTerm:
        gait_mode = gait_modes

    class FakeCommandManager:
        @staticmethod
        def get_term(name):
            assert name == "base_velocity"
            return FakeCommandTerm()

    class FakeEnv:
        scene = {"robot": FakeRobot()}
        command_manager = FakeCommandManager()

    reward = my_exoskeleton_flat.forward_velocity_reward(FakeEnv(), gait.GaitMode.RUN, scale=4.0)

    assert torch.equal(reward, torch.tensor([0.0, 0.5, 0.0]))


def test_task_speed_override_updates_selected_command_rows():
    """Imitation task speeds are per-environment and leave other rows untouched."""

    command = object.__new__(my_exoskeleton_flat.GaitVelocityCommand)
    command._env = SimpleNamespace(device=torch.device("cpu"), num_envs=3)
    command.task_speed_m_s = torch.full((3,), float("nan"))
    command.gait_mode = torch.full((3,), int(gait.GaitMode.WALK), dtype=torch.long)
    command.gait_phase = torch.zeros(3)
    command.vel_command_b = torch.zeros((3, 3))
    command.is_standing_env = torch.zeros(3, dtype=torch.bool)

    command.set_task_speed(torch.tensor([0, 2]), torch.tensor([0.55, 0.0]))

    assert torch.equal(command.vel_command_b, torch.tensor([[0.55, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert torch.equal(command.is_standing_env, torch.tensor([False, False, True]))
    assert torch.isnan(command.task_speed_m_s[1])


def _static_task_command_fake():
    """Build the command state needed to exercise task-specific gait routing."""

    command = object.__new__(my_exoskeleton_flat.GaitVelocityCommand)
    command.task_speed_m_s = torch.tensor([0.0, float("nan"), 0.55])
    command.gait_mode = torch.full((3,), int(gait.GaitMode.WALK), dtype=torch.long)
    command.gait_phase = torch.tensor([1.0, 2.0, 3.0])
    command.vel_command_b = torch.zeros(3, 3)
    command.is_standing_env = torch.zeros(3, dtype=torch.bool)
    command.previous_speed = torch.zeros(3)
    command.curriculum_phase = gait.CurriculumPhase.WALK_ONLY
    command._env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=3,
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_lin_vel_b=torch.zeros(3, 3),
                    root_ang_vel_b=torch.zeros(3, 3),
                    root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1),
                )
            ),
        }
    )
    return command


def test_static_task_remains_stand_when_curriculum_changes():
    """BT01 must keep the STAND reward route in every curriculum phase."""

    command = _static_task_command_fake()

    for phase in gait.CurriculumPhase:
        command.set_curriculum_phase(phase)
        assert command.gait_mode[0].item() == int(gait.GaitMode.STAND)
        assert command.gait_phase[0].item() == 0.0


def test_static_task_speed_sets_stand_immediately_and_resampling_preserves_it():
    """A zero-speed task must not wait for the 10 Hz mode update to stand."""

    command = _static_task_command_fake()
    command.set_task_speed(torch.tensor([0]), torch.tensor([0.0]))
    assert command.gait_mode[0].item() == int(gait.GaitMode.STAND)
    assert command.is_standing_env[0].item() is True
    assert command.vel_command_b[0, 0].item() == 0.0

    # The periodic mode update used by WALK_ONLY must not turn BT01 back into
    # WALK after a command resample.
    command._advance_mode(0.1)
    assert command.gait_mode[0].item() == int(gait.GaitMode.STAND)
