"""Gait-conditioned exoskeleton locomotion observations, commands, and rewards."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_assets.tasks.gait import (
    FROUDE_THRESHOLD,
    GAIT_MODE_COUNT,
    GRAVITY_M_S2,
    LEG_LENGTH_M,
    CurriculumPhase,
    GaitMode,
    curriculum_phase_for_step,
)
from isaaclab_assets.tasks.gait import (
    gait_mask as _shared_gait_mask,
)
from isaaclab_assets.tasks.robot_contract import (
    ACTUATED_JOINT_NAMES,
    IMU_SENSOR_NAMES,
    KNEE_ARTICULATION_LIMITS_RAD,
    KNEE_ARTICULATION_SIGNS,
    KNEE_JOINT_NAMES,
)
from isaaclab_assets.tasks.ruleresult import RuleresultRecorderMixin
from isaaclab_assets.tasks.sensor_reference import SensorReferenceBuffer

if TYPE_CHECKING:
    from isaaclab_assets.envs.my_exoskeleton_env_cfg import MyExoskeletonFlatEnvCfg


# Forces below this threshold are sensor noise rather than a foot-ground contact.
CONTACT_FORCE_THRESHOLD_N = 5.0
GAIT_UPDATE_PERIOD_S = 0.1
"""Gait-command update period [s], matching the paper's 10 Hz command layer."""

WALK_TO_STAND_HOLD_S = 1.5
RUN_TO_WALK_HOLD_S = 2.5


def _optional_sensor_enabled(env, name: str) -> bool:
    """Read an optional sensor switch while preserving legacy flat configs."""

    cfg = getattr(env, "cfg", None)
    return bool(getattr(cfg, f"use_{name}", True))


def _as_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return the Torch view exposed by either native or wrapped simulation data."""

    return value.torch if hasattr(value, "torch") else value


def _canonical_joint_tensor(env, value: torch.Tensor) -> torch.Tensor:
    """Reorder an articulation joint tensor to the public eight-joint contract."""

    value = _as_tensor(value)
    if value.ndim < 2 or value.shape[-1] != len(ACTUATED_JOINT_NAMES):
        return value
    robot = env.scene["robot"]
    names = tuple(getattr(robot, "joint_names", ()))
    if not names:
        return value
    indices = getattr(env, "_my_exoskeleton_joint_ids", None)
    if indices is None:
        missing = [name for name in ACTUATED_JOINT_NAMES if name not in names]
        if missing:
            raise ValueError(f"robot is missing canonical joints: {missing}")
        indices = torch.tensor([names.index(name) for name in ACTUATED_JOINT_NAMES], device=value.device)
        setattr(env, "_my_exoskeleton_joint_ids", indices)
    return value.index_select(-1, indices)


class MyExoskeletonFlatEnv(RuleresultRecorderMixin, ManagerBasedRLEnv):
    """Manager-based RL environment for an eight-DOF exoskeleton."""

    cfg: MyExoskeletonFlatEnvCfg

    def step(self, action: torch.Tensor):
        """Step the environment and optionally emit the structured model result."""

        result = super().step(action)
        self._ruleresult_record_after_step(result)
        return result


class GaitVelocityCommand(UniformVelocityCommand):
    """Velocity command with a five-mode gait state machine.

    Velocity targets are held for several seconds by the normal command term,
    while this state machine updates at 10 Hz. This separates the paper's
    command-layer frequency from Isaac Lab's target-resampling interval and
    leaves enough time to satisfy the 1.5 s and 2.5 s transition criteria.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Imitation environments can assign a calibrated forward speed to
        # each environment after sampling a dataset task.  ``nan`` means that
        # the ordinary curriculum/configured command ranges remain active,
        # which preserves the legacy flat-task behavior.
        self.task_speed_m_s = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.gait_mode = torch.full((self.num_envs,), int(GaitMode.WALK), dtype=torch.long, device=self.device)
        self.gait_phase = torch.rand(self.num_envs, device=self.device) * (2.0 * math.pi)
        self.transition_time = torch.zeros(self.num_envs, device=self.device)
        self.stable_time = torch.zeros(self.num_envs, device=self.device)
        self.previous_speed = torch.zeros(self.num_envs, device=self.device)
        self.curriculum_phase = CurriculumPhase.WALK_ONLY
        self._gait_update_accumulator = 0.0
        reference_path = getattr(cfg, "reference_data_path", None)
        self.reference_buffer = None
        self.reference_random_start = bool(getattr(cfg, "reference_random_start", False))
        # A directory is the multimodal recording root used by the imitation
        # task.  It is loaded by ``MyExoskeletonImitationEnv``; the legacy
        # SensorReferenceBuffer only accepts a single trajectory file.
        self.imitation_dataset_path = None
        if reference_path and Path(reference_path).is_dir():
            self.imitation_dataset_path = str(reference_path)
        elif reference_path:
            self.reference_buffer = SensorReferenceBuffer.from_file(
                reference_path,
                device=self.device,
                target_dt_s=float(getattr(cfg, "reference_target_dt_s", 0.02)),
                loop=bool(getattr(cfg, "reference_loop", True)),
            )

    def _static_task_mask(self) -> torch.Tensor:
        """Return environments pinned to a static task such as BT01.

        ``nan`` in :attr:`task_speed_m_s` means that the legacy flat task is
        using its normal command sampler.  Only finite task overrides below
        the standing threshold are treated as static; this keeps the gait
        curriculum unchanged for every ordinary walking environment.
        """

        return torch.isfinite(self.task_speed_m_s) & (self.task_speed_m_s.abs() < 0.1)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset commands and initialize gait state for the selected environments."""

        metrics = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.transition_time[ids] = 0.0
        self.stable_time[ids] = 0.0
        self.gait_phase[ids] = torch.rand_like(self.gait_phase[ids]) * (2.0 * math.pi)
        if self.reference_buffer is not None:
            self.reference_buffer.reset(ids, self.num_envs, random_start=self.reference_random_start)

        command_speed = torch.linalg.vector_norm(self.command[ids, :2], dim=-1)
        static = self._static_task_mask()
        if self.curriculum_phase == CurriculumPhase.WALK_ONLY:
            self.gait_mode[ids] = torch.where(
                static[ids],
                torch.full_like(self.gait_mode[ids], int(GaitMode.STAND)),
                torch.full_like(self.gait_mode[ids], int(GaitMode.WALK)),
            )
        else:
            initial_mode = torch.full_like(command_speed, int(GaitMode.WALK), dtype=torch.long)
            initial_mode[command_speed < 0.1] = int(GaitMode.WALK_TO_STAND)
            if self.curriculum_phase == CurriculumPhase.MULTI_GAIT:
                froude = command_speed.square() / (GRAVITY_M_S2 * LEG_LENGTH_M)
                initial_mode[froude > FROUDE_THRESHOLD] = int(GaitMode.RUN)
            initial_mode[static[ids]] = int(GaitMode.STAND)
            self.gait_mode[ids] = initial_mode

        root_speed = torch.linalg.vector_norm(_base(self._env)[0][ids, :2], dim=-1)
        self.previous_speed[ids] = root_speed
        return metrics

    def compute(self, dt: float):
        """Update the velocity command, gait phase, and transition state."""

        super().compute(dt)
        if self.reference_buffer is not None:
            # The command term is called at the policy control rate.  Keep the
            # conversion explicit so a future decimation change cannot silently
            # desynchronize the recording and the reward.
            steps = max(1, int(round(dt / float(getattr(self.cfg, "reference_target_dt_s", 0.02)))))
            self.reference_buffer.advance(steps)
        self._advance_phase(dt)
        self._gait_update_accumulator += dt
        if self._gait_update_accumulator + 1.0e-9 >= GAIT_UPDATE_PERIOD_S:
            elapsed = self._gait_update_accumulator
            self._gait_update_accumulator = 0.0
            self._advance_mode(elapsed)

    def set_curriculum_phase(self, phase: CurriculumPhase):
        """Set the active curriculum phase and remove unavailable gait labels."""

        self.curriculum_phase = CurriculumPhase(phase)
        static = self._static_task_mask()
        if self.curriculum_phase == CurriculumPhase.WALK_ONLY:
            self.gait_mode[:] = torch.where(
                static,
                torch.full_like(self.gait_mode, int(GaitMode.STAND)),
                torch.full_like(self.gait_mode, int(GaitMode.WALK)),
            )
        elif self.curriculum_phase == CurriculumPhase.TRANSITIONS:
            unavailable = (self.gait_mode == int(GaitMode.RUN)) | (self.gait_mode == int(GaitMode.RUN_TO_WALK))
            self.gait_mode[unavailable] = int(GaitMode.WALK)
        self.gait_mode[static] = int(GaitMode.STAND)
        self.gait_phase[static] = 0.0

    def set_task_speed(self, env_ids: Sequence[int] | torch.Tensor, speed_m_s: torch.Tensor | float) -> None:
        """Pin selected environments to task-specific forward speed targets.

        Args:
            env_ids: Environment indices to update.
            speed_m_s: Forward target speed [m/s], scalar or one value per ID.

        The override is applied both immediately and after later command
        resampling.  It is intentionally opt-in so the original flat task
        continues to use its curriculum ranges.
        """

        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        speeds = torch.as_tensor(speed_m_s, device=self.device, dtype=torch.float32).reshape(-1)
        if speeds.numel() == 1:
            speeds = speeds.expand(ids.numel())
        if speeds.numel() != ids.numel():
            raise ValueError("speed_m_s must be scalar or have one value per environment ID")
        if not torch.isfinite(speeds).all():
            raise ValueError("task-specific speed targets must be finite")
        self.task_speed_m_s[ids] = speeds
        self.vel_command_b[ids, 0] = speeds
        self.vel_command_b[ids, 1:] = 0.0
        static = speeds.abs() < 0.1
        self.is_standing_env[ids] = static
        self.gait_mode[ids] = torch.where(
            static,
            torch.full_like(self.gait_mode[ids], int(GaitMode.STAND)),
            torch.full_like(self.gait_mode[ids], int(GaitMode.WALK)),
        )
        self.gait_phase[ids] = torch.where(
            static, torch.zeros_like(self.gait_phase[ids]), self.gait_phase[ids]
        )

    def clear_task_speed(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Restore configured/curriculum command sampling for selected IDs."""

        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel():
            self.task_speed_m_s[ids] = float("nan")
            self.gait_mode[ids] = int(GaitMode.WALK)

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample commands, then reapply any imitation task speed overrides."""

        if isinstance(env_ids, slice):
            start, stop, step = env_ids.indices(self.num_envs)
            ids = torch.arange(start, stop, step, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        super()._resample_command(ids)
        speeds = self.task_speed_m_s[ids]
        valid = torch.isfinite(speeds)
        if valid.any():
            selected = ids[valid]
            self.vel_command_b[selected, 0] = speeds[valid]
            self.vel_command_b[selected, 1:] = 0.0
            selected_static = speeds[valid].abs() < 0.1
            self.is_standing_env[selected] = selected_static
            self.gait_mode[selected] = torch.where(
                selected_static,
                torch.full_like(self.gait_mode[selected], int(GaitMode.STAND)),
                torch.full_like(self.gait_mode[selected], int(GaitMode.WALK)),
            )
            self.gait_phase[selected] = torch.where(
                selected_static, torch.zeros_like(self.gait_phase[selected]), self.gait_phase[selected]
            )

    def _resample(self, env_ids: Sequence[int]):
        """Normalize a full-slice reset before the base resampling helper."""

        if isinstance(env_ids, slice):
            start, stop, step = env_ids.indices(self.num_envs)
            env_ids = torch.arange(start, stop, step, device=self.device, dtype=torch.long)
        super()._resample(env_ids)

    def _advance_phase(self, dt: float):
        cadence = torch.zeros(self.num_envs, device=self.device)
        cadence[self.gait_mode == int(GaitMode.WALK)] = 0.8
        cadence[self.gait_mode == int(GaitMode.WALK_TO_STAND)] = 0.6
        cadence[self.gait_mode == int(GaitMode.RUN)] = 1.6
        cadence[self.gait_mode == int(GaitMode.RUN_TO_WALK)] = 1.0
        self.gait_phase.add_(2.0 * math.pi * cadence * dt).remainder_(2.0 * math.pi)
        self.gait_phase[self.gait_mode == int(GaitMode.STAND)] = 0.0

    def _advance_mode(self, dt: float):
        static = self._static_task_mask()
        if self.curriculum_phase == CurriculumPhase.WALK_ONLY:
            self.gait_mode[:] = torch.where(
                static,
                torch.full_like(self.gait_mode, int(GaitMode.STAND)),
                torch.full_like(self.gait_mode, int(GaitMode.WALK)),
            )
            self.gait_phase[static] = 0.0
            self.previous_speed[:] = torch.linalg.vector_norm(_base(self._env)[0][:, :2], dim=-1)
            return

        command_speed = torch.linalg.vector_norm(self.command[:, :2], dim=-1)
        root_speed = torch.linalg.vector_norm(_base(self._env)[0][:, :2], dim=-1)
        left_force, right_force = _foot_contacts(self._env)
        double_support = (left_force > CONTACT_FORCE_THRESHOLD_N) & (right_force > CONTACT_FORCE_THRESHOLD_N)
        low_command = (command_speed < 0.1) & ~static

        entering_stop = (
            low_command & (self.gait_mode != int(GaitMode.STAND)) & (self.gait_mode != int(GaitMode.WALK_TO_STAND))
        )
        self.gait_mode[entering_stop] = int(GaitMode.WALK_TO_STAND)
        self.transition_time[entering_stop] = 0.0
        self.stable_time[entering_stop] = 0.0

        stopping = low_command & (self.gait_mode == int(GaitMode.WALK_TO_STAND))
        stable_stop = stopping & double_support & (root_speed < 0.1)
        self.stable_time[stopping] = torch.where(
            stable_stop[stopping], self.stable_time[stopping] + dt, torch.zeros_like(self.stable_time[stopping])
        )
        stopped = stopping & (self.stable_time >= WALK_TO_STAND_HOLD_S)
        self.gait_mode[stopped] = int(GaitMode.STAND)
        self.gait_phase[stopped] = 0.0

        moving_command = (~low_command) & ~static
        if self.curriculum_phase == CurriculumPhase.TRANSITIONS:
            self.gait_mode[moving_command] = int(GaitMode.WALK)
        else:
            froude = command_speed.square() / (GRAVITY_M_S2 * LEG_LENGTH_M)
            run_command = moving_command & (froude > FROUDE_THRESHOLD)
            self.gait_mode[run_command] = int(GaitMode.RUN)

            walk_command = moving_command & ~run_command
            entering_r2w = walk_command & (self.gait_mode == int(GaitMode.RUN))
            self.gait_mode[entering_r2w] = int(GaitMode.RUN_TO_WALK)
            self.transition_time[entering_r2w] = 0.0

            transitioning = walk_command & (self.gait_mode == int(GaitMode.RUN_TO_WALK))
            self.transition_time[transitioning] += dt
            completed = transitioning & (self.transition_time >= RUN_TO_WALK_HOLD_S)
            self.gait_mode[completed] = int(GaitMode.WALK)

            regular_walk = walk_command & ~transitioning & ~entering_r2w
            self.gait_mode[regular_walk] = int(GaitMode.WALK)

        self.transition_time[self.gait_mode == int(GaitMode.WALK_TO_STAND)] += dt
        self.gait_mode[static] = int(GaitMode.STAND)
        self.gait_phase[static] = 0.0
        self.transition_time[static] = 0.0
        self.stable_time[static] = 0.0
        self.previous_speed[:] = root_speed

    @property
    def reference_sample(self) -> dict[str, torch.Tensor] | None:
        """Return the current sensor reference for each environment, if enabled."""

        return None if self.reference_buffer is None else self.reference_buffer.sample()


def _sensor_stats(sensor, env_count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normal-force sum, mean, and standard deviation per environment."""
    try:
        force_value = sensor.data.net_normal_forces_w
    except AttributeError as exc:
        raise RuntimeError("foot ContactSensor must expose net_normal_forces_w") from exc
    if force_value is None:
        raise RuntimeError("foot ContactSensor returned no net_normal_forces_w buffer")
    force = _as_tensor(force_value)
    if force.ndim != 3 or force.shape[0] != env_count or force.shape[-1] != 3:
        raise ValueError(
            "foot ContactSensor net_normal_forces_w must have shape "
            f"({env_count}, num_sensors, 3), got {tuple(force.shape)}"
        )
    if force.device != device:
        force = force.to(device=device)
    magnitudes = force.reshape(env_count, -1, 3).norm(dim=-1)
    return magnitudes.sum(dim=-1), magnitudes.mean(dim=-1), magnitudes.std(dim=-1, unbiased=False)


def _foot_contacts(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Read left/right ContactSensor net forces and convert them to contact magnitudes."""
    if not _optional_sensor_enabled(env, "contact"):
        zeros = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        return zeros, zeros.clone()
    left = _sensor_stats(env.scene["left_foot_contact"], env.num_envs, env.device)[0]
    right = _sensor_stats(env.scene["right_foot_contact"], env.num_envs, env.device)[0]
    return left, right


def _foot_pressure_features(env) -> torch.Tensor:
    """Return [left mean, left std, right mean, right std] contact features."""
    if not _optional_sensor_enabled(env, "contact"):
        return torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
    left = _sensor_stats(env.scene["left_foot_contact"], env.num_envs, env.device)
    right = _sensor_stats(env.scene["right_foot_contact"], env.num_envs, env.device)
    return torch.stack((left[1], left[2], right[1], right[2]), dim=-1)


def _reference_sample(env) -> dict[str, torch.Tensor] | None:
    """Return the command-owned reference sample, if a recording is enabled."""

    try:
        sample = _gait_command(env).reference_sample
    except (AttributeError, TypeError):
        return None
    return sample


def _imu_gyro_features(env) -> torch.Tensor:
    """Return six link-IMU gyro vectors in the canonical sensor order.

    The fallback repeats the articulated root gyro.  It keeps unit tests and
    old scenes usable, but a real imitation run should provide all six IMUs so
    that the fallback is never used.
    """

    root_gyro = _base(env)[1]
    if not _optional_sensor_enabled(env, "imu"):
        return torch.zeros(
            (root_gyro.shape[0], len(getattr(env, "imitation_imu_sensor_names", IMU_SENSOR_NAMES)), 3),
            device=root_gyro.device,
            dtype=root_gyro.dtype,
        )
    # The imitation task exposes recorder IDs (imu_01..imu_06), while the
    # legacy flat task keeps the original anatomical aliases.  Prefer the
    # environment's explicit contract and retain the root-gyro fallback for
    # scenes that do not instantiate link IMUs.
    sensor_names = tuple(getattr(env, "imitation_imu_sensor_names", IMU_SENSOR_NAMES))
    readings = []
    for sensor_name in sensor_names:
        try:
            value = _as_tensor(env.scene[sensor_name].data.ang_vel_b)
        except (AttributeError, KeyError, TypeError):
            return root_gyro[:, None, :].expand(-1, len(sensor_names), -1)
        if value.ndim == 3:
            value = value[:, 0, :]
        if value.ndim != 2 or value.shape[-1] != 3 or value.shape[0] != root_gyro.shape[0]:
            raise ValueError(f"IMU {sensor_name} angular velocity must have shape [N, 3] or [N, 1, 3]")
        readings.append(value)
    return torch.stack(readings, dim=1)


def sensor_reference_imitation_reward(
    env,
    joint_pos_weight: float = 1.0,
    joint_vel_weight: float = 0.25,
    gyro_weight: float = 0.5,
    contact_weight: float = 0.25,
    joint_pos_sigma: float = 0.35,
    joint_vel_sigma: float = 1.5,
    gyro_sigma: float = 1.0,
    contact_sigma: float = 0.35,
):
    """Score agreement with an optional recorded sensor trajectory.

    The reference is sampled by :class:`GaitVelocityCommand` at the policy
    rate.  Missing optional channels are omitted and the remaining weights are
    renormalized, so a joint-angle-only recording is valid.  This term is a
    training signal; it must not be enabled with a recording whose joint order,
    zero offsets, or sensor frames have not been calibrated.
    """

    weights = (joint_pos_weight, joint_vel_weight, gyro_weight, contact_weight)
    if any(weight < 0.0 for weight in weights) or not any(weight > 0.0 for weight in weights):
        raise ValueError("imitation weights must be non-negative and contain at least one positive value")
    sigmas = (joint_pos_sigma, joint_vel_sigma, gyro_sigma, contact_sigma)
    if any(sigma <= 0.0 for sigma in sigmas):
        raise ValueError("imitation sigmas must be positive")

    joint_pos = _canonical_joint_tensor(env, env.scene["robot"].data.joint_pos)
    zero = torch.zeros(joint_pos.shape[0], device=joint_pos.device, dtype=joint_pos.dtype)
    sample = _reference_sample(env)
    if sample is None:
        return zero

    scores = []
    active_weights = []
    reference_joint_pos = sample["joint_pos_rad"].to(dtype=joint_pos.dtype)
    scores.append(torch.exp(-torch.mean(torch.square((joint_pos - reference_joint_pos) / joint_pos_sigma), dim=-1)))
    active_weights.append(joint_pos_weight)

    if joint_vel_weight > 0.0 and "joint_vel_rad_s" in sample:
        joint_vel = _canonical_joint_tensor(env, env.scene["robot"].data.joint_vel)
        reference_joint_vel = sample["joint_vel_rad_s"].to(dtype=joint_vel.dtype)
        scores.append(torch.exp(-torch.mean(torch.square((joint_vel - reference_joint_vel) / joint_vel_sigma), dim=-1)))
        active_weights.append(joint_vel_weight)

    if gyro_weight > 0.0 and "gyro_rad_s" in sample and _optional_sensor_enabled(env, "imu"):
        gyro = _imu_gyro_features(env)
        reference_gyro = sample["gyro_rad_s"].to(dtype=gyro.dtype)
        scores.append(torch.exp(-torch.mean(torch.square((gyro - reference_gyro) / gyro_sigma), dim=(-1, -2))))
        active_weights.append(gyro_weight)

    if contact_weight > 0.0 and "contact_lr" in sample and _optional_sensor_enabled(env, "contact"):
        left, right = _foot_contacts(env)
        contact = torch.stack((left > CONTACT_FORCE_THRESHOLD_N, right > CONTACT_FORCE_THRESHOLD_N), dim=-1).to(
            joint_pos.dtype
        )
        reference_contact = sample["contact_lr"].to(dtype=contact.dtype)
        scores.append(torch.exp(-torch.mean(torch.square((contact - reference_contact) / contact_sigma), dim=-1)))
        active_weights.append(contact_weight)

    # Optional IMU/contact switches can remove every reference channel except
    # the mandatory joint stream.  Be defensive for a caller that also sets
    # the joint weights to zero: a disabled imitation term must be finite and
    # neutral rather than producing a 0/0 NaN.
    if not active_weights:
        return zero
    total_weight = sum(active_weights)
    score = sum(weight * value for weight, value in zip(active_weights, scores, strict=True)) / total_weight
    if "gait_id" in sample:
        gait = _gait_command(env)
        reference_gait = sample["gait_id"].to(device=joint_pos.device)
        # A negative label means "all modes" and is useful for recordings
        # without a synchronized gait classifier.
        valid = (reference_gait < 0) | (reference_gait == gait.gait_mode)
        score = score * valid.to(dtype=score.dtype)
    return score


def exoskeleton_observation(env) -> torch.Tensor:
    """Build the 49-D gait-conditioned policy vector.

    Layout: projected gravity (3), base angular/linear velocity (3+3), joint
    position/velocity/previous normalized action (8+8+8), left/right contact
    magnitudes (1+1), left/right pressure mean and standard deviation (4), and
    velocity command (3), gait phase sine/cosine (2), and gait one-hot (5).
    ContactSensor exposes aggregate forces rather than Gazebo's 60 bins; the
    aggregate is used as a stable pressure mean/std proxy.
    """
    robot = env.scene["robot"]
    data = robot.data
    gravity = _as_tensor(data.projected_gravity_b)
    ang = _as_tensor(data.root_ang_vel_b)
    lin = _as_tensor(data.root_lin_vel_b)
    pos = _canonical_joint_tensor(env, data.joint_pos)
    vel = _canonical_joint_tensor(env, data.joint_vel)
    # The paper exposes the previous policy action.  ``prev_action`` is the
    # normalized action sent to the manager, not a position target; adding it
    # to the current joint position would therefore mix radians and normalized
    # units and would describe neither the controller input nor its target.
    action_manager = getattr(env, "action_manager", None)
    previous_action = (
        torch.zeros_like(pos) if action_manager is None else _as_tensor(getattr(action_manager, "prev_action"))
    )
    if previous_action.ndim != 2 or previous_action.shape != pos.shape:
        raise ValueError(
            "my_exoskeleton expects one 8-DOF previous action, "
            f"got {tuple(previous_action.shape)} for joint positions {tuple(pos.shape)}"
        )
    left, right = _foot_contacts(env)
    commands = env.command_manager.get_command("base_velocity")
    pressure = _foot_pressure_features(env)
    gait = _gait_command(env)
    phase = torch.stack((torch.sin(gait.gait_phase), torch.cos(gait.gait_phase)), dim=-1)
    gait_one_hot = torch.nn.functional.one_hot(gait.gait_mode, num_classes=GAIT_MODE_COUNT).to(pos.dtype)
    return torch.cat(
        (
            gravity,
            ang,
            lin,
            pos,
            vel,
            previous_action,
            left[:, None],
            right[:, None],
            pressure,
            commands,
            phase,
            gait_one_hot,
        ),
        dim=-1,
    )


def exoskeleton_critic_observation(env) -> torch.Tensor:
    """Build privileged critic observations with contact and dynamics state."""

    policy = exoskeleton_observation(env)
    robot = env.scene["robot"]
    left, right = _foot_contacts(env)
    contacts = torch.stack((left > CONTACT_FORCE_THRESHOLD_N, right > CONTACT_FORCE_THRESHOLD_N), dim=-1).to(
        policy.dtype
    )
    forces = torch.stack((left, right), dim=-1) / 500.0
    root_height = _as_tensor(robot.data.root_pos_w)[:, 2:3]
    joint_acc = _canonical_joint_tensor(env, robot.data.joint_acc)
    return torch.cat((policy, contacts, forces, root_height, joint_acc), dim=-1)


def _base(env):
    """Return root state tensors in the simulation device."""
    data = env.scene["robot"].data
    values = (data.root_lin_vel_b, data.root_ang_vel_b, data.root_quat_w)
    return tuple(_as_tensor(value) for value in values)


def _gait_command(env) -> GaitVelocityCommand:
    """Return the gait-aware base velocity command term."""

    term = env.command_manager.get_term("base_velocity")
    if not hasattr(term, "gait_mode"):
        raise TypeError("base_velocity must use GaitVelocityCommand")
    return term


def gait_mask(env, gait: GaitMode | int) -> torch.Tensor:
    """Return a per-environment float mask for one gait label."""

    return _shared_gait_mask(env, gait)


def _masked(value: torch.Tensor, env, gait: GaitMode | int) -> torch.Tensor:
    return value * gait_mask(env, gait).to(value.dtype)


def _locomotion_mask(env) -> torch.Tensor:
    """Return one for walking, running, and transition gaits.

    A few shared biomechanical terms should not reward leg motion while the
    robot is in the static ``STAND`` mode.  Keeping this helper tolerant of
    small unit-test fakes also makes the reward functions usable before the
    command manager has been constructed.
    """

    try:
        command = _gait_command(env)
    except (AttributeError, TypeError):
        return torch.ones(env.scene["robot"].data.joint_vel.shape[0], device=env.scene["robot"].data.joint_vel.device)
    modes = command.gait_mode
    return (
        (modes == int(GaitMode.WALK))
        | (modes == int(GaitMode.WALK_TO_STAND))
        | (modes == int(GaitMode.RUN))
        | (modes == int(GaitMode.RUN_TO_WALK))
    ).to(dtype=torch.float32)


def _forward_motion_gate(env, speed_scale: float = 0.2) -> torch.Tensor:
    """Return a bounded gate that is zero for a stationary policy."""

    if speed_scale <= 0.0:
        raise ValueError("speed_scale must be positive")
    forward_speed = _base(env)[0][:, 0]
    return torch.clamp(forward_speed / speed_scale, min=0.0, max=1.0)


def tracking_reward(env):
    """Reward exponential tracking of commanded planar velocity and yaw rate."""
    lin, ang, _ = _base(env)
    cmd = env.command_manager.get_command("base_velocity")
    error = torch.cat((lin[:, :2] - cmd[:, :2], ang[:, 2:3] - cmd[:, 2:3]), dim=-1)
    return torch.exp(-torch.sum(error.square(), dim=-1) / 0.25)


def tracking_linear_velocity_reward(
    env,
    std: float = 0.5,
    activity_gate_speed: float | None = None,
    activity_gate_command_max: float | None = None,
):
    """Reward planar command tracking, optionally requiring low-speed forward activity.

    The low-speed gate is used only for the fixed forward curriculum.  It removes
    the stationary ``exp(-0.2**2 / std**2)`` shortcut without making the higher
    speed command distribution unnecessarily sharp.
    """

    lin, _, _ = _base(env)
    command = env.command_manager.get_command("base_velocity")
    error = torch.sum(torch.square(lin[:, :2] - command[:, :2]), dim=-1)
    score = torch.exp(-error / (std * std))
    if activity_gate_speed is not None or activity_gate_command_max is not None:
        if activity_gate_speed is None or activity_gate_command_max is None:
            raise ValueError("activity gate requires both speed and command thresholds")
        if activity_gate_speed <= 0.0 or activity_gate_command_max <= 0.0:
            raise ValueError("activity gate thresholds must be positive")
        low_forward_command = (command[:, 0] > 0.0) & (
            torch.linalg.vector_norm(command[:, :2], dim=-1) <= activity_gate_command_max
        )
        activity = torch.clamp(lin[:, 0] / activity_gate_speed, min=0.0, max=1.0)
        score = torch.where(low_forward_command, score * activity, score)
    return score


def tracking_yaw_velocity_reward(env, std: float = 0.5):
    """Reward yaw-rate command tracking with an exponential kernel."""

    _, ang, _ = _base(env)
    command = env.command_manager.get_command("base_velocity")
    return torch.exp(-torch.square(ang[:, 2] - command[:, 2]) / (std * std))


def fixed_forward_walk_reward(
    env,
    target_speed: float,
    speed_tolerance: float,
    lateral_speed_std: float,
    yaw_rate_std: float,
    command_speed_tolerance: float | None = None,
    gait: GaitMode | int | None = None,
):
    """Reward walking at a target velocity along the robot's body-frame positive x-axis.

    The speed window is deliberately zero at rest, reverse motion, and twice
    the target speed.  This prevents the posture and contact terms from making
    a stationary policy competitive with a walking policy.

    Args:
        env: The manager-based reinforcement-learning environment.
        target_speed: Desired body-forward velocity [m/s]. Must be positive.
        speed_tolerance: Half-width of the quadratic forward-speed window [m/s].
        lateral_speed_std: Gaussian decay scale for body lateral velocity [m/s].
        yaw_rate_std: Gaussian decay scale for body yaw velocity [rad/s].
        command_speed_tolerance: Optional command-speed gate [m/s]. When set,
            the term is faded out for later curriculum commands whose speed is
            no longer the fixed walking target.
        gait: Optional gait label used to route the term to one gait.

    Returns:
        A reward in ``[0, 1]`` for each environment.
    """
    lin_vel_b, ang_vel_b, _ = _base(env)
    forward_speed = lin_vel_b[:, 0]
    forward_score = torch.clamp(1.0 - torch.square((forward_speed - target_speed) / speed_tolerance), min=0.0, max=1.0)
    straight_score = torch.exp(
        -torch.square(lin_vel_b[:, 1] / lateral_speed_std) - torch.square(ang_vel_b[:, 2] / yaw_rate_std)
    )
    score = forward_score * straight_score
    if command_speed_tolerance is not None and hasattr(env, "command_manager"):
        command = env.command_manager.get_command("base_velocity")
        if command_speed_tolerance <= 0.0:
            raise ValueError("command_speed_tolerance must be positive")
        command_gate = torch.exp(-torch.square((command[:, 0] - target_speed) / command_speed_tolerance))
        score = score * command_gate
    if gait is not None and hasattr(env, "command_manager"):
        score = _masked(score, env, gait)
    return score


def lin_vel_z_reward(env):
    """Return vertical-speed magnitude for the configured negative weight."""
    return torch.abs(_base(env)[0][:, 2])


def ang_vel_xy_reward(env):
    """Return roll/pitch angular-speed magnitude for the configured negative weight."""
    return torch.abs(_base(env)[1][:, :2]).sum(dim=-1)


def orientation_reward(env):
    """Return uprightness score based on roll and pitch."""
    roll, pitch, _ = euler_xyz_from_quat(_base(env)[2])
    return 1.0 - torch.abs(roll) - torch.abs(pitch)


def torque_reward(env):
    """Return squared joint effort; RewardTermCfg supplies -1e-6."""
    effort = _as_tensor(env.scene["robot"].data.applied_torque)
    return effort.square().sum(dim=-1)


def action_rate_reward(env):
    """Return squared action change; RewardTermCfg supplies -0.1."""
    action = env.action_manager.action
    previous = env.action_manager.prev_action
    return (action - previous).square().sum(dim=-1)


def contact_reward(env):
    """Reward having at least one foot in contact without rewarding impact magnitude.

    Contact sensors report forces in newtons.  Returning the raw force here would make a
    hard landing worth orders of magnitude more than velocity tracking, so this term is
    deliberately bounded to ``[0, 1]``.
    """
    left, right = _foot_contacts(env)
    return ((left > CONTACT_FORCE_THRESHOLD_N) | (right > CONTACT_FORCE_THRESHOLD_N)).float()


def foot_contact_alternation_reward(env):
    """Reward exclusive left/right contact, encouraging alternating gait phases."""
    left, right = _foot_contacts(env)
    left_on, right_on = left > CONTACT_FORCE_THRESHOLD_N, right > CONTACT_FORCE_THRESHOLD_N
    return (left_on ^ right_on).float()


def feet_forward_alignment_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    sigma: float,
    min_forward_speed: float,
    max_lateral_speed: float,
    max_yaw_rate: float,
):
    """Reward both feet facing the commanded forward direction during straight walking.

    The velocity command is expressed in the base frame.  For a positive
    straight-ahead command, its world-frame direction is therefore the root
    yaw direction.  Only yaw is compared: the left/right foot links have
    mirrored roll frames even while their forward axes are parallel.
    """
    robot = env.scene[asset_cfg.name]
    foot_quat = _as_tensor(robot.data.body_link_quat_w)
    foot_quat = foot_quat[:, asset_cfg.body_ids]

    root_quat = _as_tensor(robot.data.root_quat_w)

    foot_yaw = math_utils.yaw_quat(foot_quat)
    desired_yaw = math_utils.yaw_quat(root_quat).unsqueeze(1).expand_as(foot_yaw)
    relative_yaw = math_utils.quat_mul(math_utils.quat_conjugate(desired_yaw), foot_yaw)
    yaw_error = relative_yaw[..., :3].square().sum(dim=-1)
    alignment = torch.exp(-yaw_error / (2.0 * sigma * sigma)).mean(dim=1)

    command = env.command_manager.get_command(command_name)
    is_straight_forward = (
        (command[:, 0] > min_forward_speed)
        & (command[:, 1].abs() <= max_lateral_speed)
        & (command[:, 2].abs() <= max_yaw_rate)
    )
    return alignment * is_straight_forward.float()


def _centroidal_angular_momentum_z(data) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute total and side-split centroidal angular momentum about world ``z``.

    Returns:
        A tuple containing total ``z`` angular momentum and the per-body
        momentum vectors in ``[kg m^2/s]``.  The latter is used only by the
        morphology-adapted replacement for the paper's arm term.
    """

    mass = _as_tensor(data.body_mass)
    com_pos = _as_tensor(data.body_com_pos_w)
    com_vel = _as_tensor(data.body_com_vel_w)
    inertia = _as_tensor(data.body_inertia)
    body_link_quat_w = _as_tensor(data.body_link_quat_w)
    if mass.ndim != 2 or com_pos.ndim != 3 or com_vel.ndim != 3 or inertia.ndim != 3:
        raise ValueError("unexpected articulation body-state shapes for centroidal momentum")
    if (
        com_vel.shape[-1] < 6
        or inertia.shape[-1] != 9
        or body_link_quat_w.shape[-1] != 4
        or mass.shape[:2] != com_pos.shape[:2]
        or body_link_quat_w.shape[:2] != mass.shape[:2]
    ):
        raise ValueError("incomplete articulation body state for centroidal momentum")

    mass = mass.clamp_min(1.0e-8)
    total_mass = mass.sum(dim=1, keepdim=True)
    centroid = (mass.unsqueeze(-1) * com_pos).sum(dim=1, keepdim=True) / total_mass.unsqueeze(-1)
    centroid_vel = (mass.unsqueeze(-1) * com_vel[..., :3]).sum(dim=1, keepdim=True) / total_mass.unsqueeze(-1)
    relative_pos = com_pos - centroid
    relative_vel = com_vel[..., :3] - centroid_vel
    orbital = torch.cross(relative_pos, mass.unsqueeze(-1) * relative_vel, dim=-1)
    # PhysX returns COM-centered inertia in the rigid-body-prim frame as nine
    # column-major values.  Convert to a row-major matrix and rotate it into
    # world coordinates before multiplying by world angular velocity.
    inertia_link = inertia.reshape(inertia.shape[0], inertia.shape[1], 3, 3).transpose(-1, -2)
    link_rotation_w = math_utils.matrix_from_quat(body_link_quat_w)
    inertia_world = link_rotation_w @ inertia_link @ link_rotation_w.transpose(-1, -2)
    spin = torch.einsum("nbij,nbj->nbi", inertia_world, com_vel[..., 3:6])
    body_momentum = orbital + spin
    total = body_momentum.sum(dim=1)[:, 2]
    return total, body_momentum


def _side_momentum_z(body_momentum: torch.Tensor, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Split selected ordered body momenta into right and left halves."""

    ids = asset_cfg.body_ids
    selected = body_momentum if isinstance(ids, slice) else body_momentum[:, ids]
    if selected.shape[1] < 2 or selected.shape[1] % 2:
        raise ValueError("side momentum requires an even, ordered body selection")
    midpoint = selected.shape[1] // 2
    right = selected[:, :midpoint, 2].sum(dim=-1)
    left = selected[:, midpoint:, 2].sum(dim=-1)
    return left, right


def leg_momentum_balance_reward(
    env,
    asset_cfg: SceneEntityCfg,
    momentum_scale: float = 10.0,
):
    """Implement the paper's angular-momentum cost for the legs-only robot.

    For the humanoid in the paper, Eq. (2) contains a total angular-momentum
    cost and a mirrored left/right arm residual.  In this legs-only model the
    adapted residual is ``(-L_left,z - L_right,z)`` in the shared world frame.

    The exoskeleton has no arms, so the second term uses ordered left/right
    leg-body momentum as a transparent morphology-adapted surrogate.  The
    total term is computed from the actual Isaac Lab body masses, COM
    velocities, COM positions, and world-frame inertia tensors.
    """

    if momentum_scale <= 0.0:
        raise ValueError("momentum_scale must be positive")
    data = env.scene[asset_cfg.name].data
    total_z, body_momentum = _centroidal_angular_momentum_z(data)
    left_z, right_z = _side_momentum_z(body_momentum, asset_cfg)
    # The paper writes the side term in mirrored local arm coordinates.  A
    # sagittal mirror flips the left-side yaw sign, so its equivalent in the
    # shared world frame is ``(-L_left,z - L_right,z)^2``: counter-rotating
    # legs cancel instead of being penalized as excessive activity.
    mirrored_side_residual = -left_z - right_z
    residual = torch.square(total_z / momentum_scale) + 0.4 * torch.square(mirrored_side_residual / momentum_scale)
    score = -residual
    return score * _locomotion_mask(env).to(score.dtype)


def leg_mechanical_power_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    power_scale: float = 50.0,
):
    """Penalize normalized lower-limb mechanical power [W].

    The paper's arm-swing energy term cannot be evaluated on this eight-DOF
    mechanism.  ``|tau * qdot|`` over the actuated leg joints is its explicit
    lower-body energy surrogate, routed only to moving gaits so that standing
    support torque is not mistaken for inefficient swing.
    """

    if power_scale <= 0.0:
        raise ValueError("power_scale must be positive")
    data = env.scene[asset_cfg.name].data
    joint_vel = _as_tensor(data.joint_vel)[:, asset_cfg.joint_ids]
    effort = getattr(data, "applied_torque", None)
    if effort is None:
        effort = getattr(data, "computed_torque", None)
    if effort is None:
        effort = torch.zeros_like(_as_tensor(data.joint_vel))
    effort = _as_tensor(effort)[:, asset_cfg.joint_ids]
    power = torch.nan_to_num(torch.abs(effort * joint_vel), nan=0.0, posinf=power_scale, neginf=power_scale)
    normalized = torch.clamp(power.mean(dim=-1) / power_scale, min=0.0, max=10.0)
    return normalized * _locomotion_mask(env).to(normalized.dtype)


def leg_phase_symmetry_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sigma: float = 0.5,
    activity_scale: float = 0.5,
    motion_scale: float = 0.2,
):
    """Reward anti-phase left/right leg motion, replacing arm symmetry."""

    if sigma <= 0.0 or activity_scale <= 0.0:
        raise ValueError("sigma and activity_scale must be positive")
    joint_vel = _as_tensor(env.scene[asset_cfg.name].data.joint_vel)[:, asset_cfg.joint_ids]
    if joint_vel.shape[-1] != 8:
        raise ValueError("leg_phase_symmetry_reward requires four ordered joints per leg")
    pair_error = torch.mean(torch.square(joint_vel[:, :4] + joint_vel[:, 4:8]), dim=-1)
    activity = torch.tanh(torch.mean(torch.abs(joint_vel), dim=-1) / activity_scale)
    score = activity * torch.exp(-pair_error / (sigma * sigma))
    return score * _forward_motion_gate(env, motion_scale) * _locomotion_mask(env).to(score.dtype)


def leg_amplitude_match_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sigma: float = 0.5,
    activity_scale: float = 0.5,
    motion_scale: float = 0.2,
):
    """Reward comparable left/right swing amplitudes without arm joints."""

    if sigma <= 0.0 or activity_scale <= 0.0:
        raise ValueError("sigma and activity_scale must be positive")
    joint_vel = _as_tensor(env.scene[asset_cfg.name].data.joint_vel)[:, asset_cfg.joint_ids]
    if joint_vel.shape[-1] != 8:
        raise ValueError("leg_amplitude_match_reward requires four ordered joints per leg")
    right_amp = torch.linalg.vector_norm(joint_vel[:, :4], dim=-1) / 2.0
    left_amp = torch.linalg.vector_norm(joint_vel[:, 4:8], dim=-1) / 2.0
    activity = torch.tanh((right_amp + left_amp) / (2.0 * activity_scale))
    score = activity * torch.exp(-torch.square(right_amp - left_amp) / (sigma * sigma))
    return score * _forward_motion_gate(env, motion_scale) * _locomotion_mask(env).to(score.dtype)


def phase_contact_reward(
    env,
    gait: GaitMode | int,
    allow_double_support: bool = False,
    flight_window: float = 0.0,
    double_support_ramp_s: float = 0.0,
):
    """Reward the contact pattern implied by the gait phase.

    Walking keeps an exact single-support target.  A transition can opt into
    double support, while running can reserve a small phase window for flight
    (no foot contact), matching the paper's alternating support/flight pattern.
    ``flight_window`` is a threshold on ``abs(sin(gait_phase))`` and must be in
    ``[0, 1)``.  When ``double_support_ramp_s`` is positive, double-support
    credit ramps from zero to one over that gait transition duration.
    """

    if not 0.0 <= flight_window < 1.0:
        raise ValueError("flight_window must be in [0, 1)")
    if double_support_ramp_s < 0.0:
        raise ValueError("double_support_ramp_s must be non-negative")

    left, right = _foot_contacts(env)
    observed = torch.stack((left > CONTACT_FORCE_THRESHOLD_N, right > CONTACT_FORCE_THRESHOLD_N), dim=-1)
    phase = _gait_command(env).gait_phase
    phase_sine = torch.sin(phase)
    expected_left = phase_sine >= 0.0
    expected = torch.stack((expected_left, ~expected_left), dim=-1)
    single_support_match = torch.all(observed == expected, dim=-1).float()
    # W2S may need a brief double-support landing to reshape contact timing.
    # Walk remains exact single support by leaving this disabled.
    score = single_support_match
    if allow_double_support:
        double_support = torch.all(observed, dim=-1).to(score.dtype)
        if double_support_ramp_s > 0.0:
            transition_time = _gait_command(env).transition_time
            blend = torch.clamp(transition_time / double_support_ramp_s, min=0.0, max=1.0).to(score.dtype)
            score = score * (1.0 - blend) + double_support * blend
        else:
            score = torch.maximum(score, double_support)
    if flight_window > 0.0:
        # Run has a bounded flight interval around each support exchange.  Do
        # not allow flight for other gaits merely because a caller typoed the
        # optional parameter.
        if int(gait) != int(GaitMode.RUN):
            raise ValueError("flight_window is only valid for the RUN gait")
        flight_phase = phase_sine.abs() < flight_window
        flight_match = (~torch.any(observed, dim=-1)).to(score.dtype)
        score = torch.where(flight_phase, flight_match, score)
    # A mean would give double support a constant 0.5 reward even though the
    # walking target is single support. Exact matching avoids that shortcut.
    return _masked(score, env, gait)


def swing_height_penalty(
    env,
    gait: GaitMode | int,
    asset_cfg: SceneEntityCfg,
    target_height: float,
):
    """Penalize swing-foot height error [m^2] for one gait."""

    foot_pos = _as_tensor(env.scene[asset_cfg.name].data.body_link_pos_w)[:, asset_cfg.body_ids]
    left, right = _foot_contacts(env)
    airborne = torch.stack((left <= CONTACT_FORCE_THRESHOLD_N, right <= CONTACT_FORCE_THRESHOLD_N), dim=-1).float()
    error = torch.square(foot_pos[:, :, 2] - target_height) * airborne
    normalizer = airborne.sum(dim=-1).clamp(min=1.0)
    return _masked(error.sum(dim=-1) / normalizer, env, gait)


def straight_knee_reward(env, gait: GaitMode | int, asset_cfg: SceneEntityCfg, sigma: float = 0.35):
    """Reward straight knees on the stance leg during the selected gait."""

    joint_pos = _as_tensor(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids]
    if joint_pos.shape[-1] != 2:
        raise ValueError("straight_knee_reward requires ordered right and left knee joints")
    left_force, right_force = _foot_contacts(env)
    stance = torch.stack((right_force > CONTACT_FORCE_THRESHOLD_N, left_force > CONTACT_FORCE_THRESHOLD_N), dim=-1).to(
        joint_pos.dtype
    )
    signs = torch.as_tensor(
        [KNEE_ARTICULATION_SIGNS[name] for name in KNEE_JOINT_NAMES],
        device=joint_pos.device,
        dtype=joint_pos.dtype,
    )
    flexion = joint_pos * signs
    normalizer = stance.sum(dim=-1).clamp(min=1.0)
    error = (torch.square(flexion) * stance).sum(dim=-1) / normalizer
    score = torch.exp(-error / (sigma * sigma)) * (stance.sum(dim=-1) > 0.0).to(joint_pos.dtype)
    return _masked(score, env, gait)


def _knee_joint_names(asset_cfg: SceneEntityCfg) -> tuple[str, ...]:
    """Return the explicit knee order used by a manager term.

    Isaac Lab resolves ``joint_ids`` during scene construction, but preserving
    the names here lets safety terms apply side-specific bounds even when the
    backend's internal joint order changes.
    """

    names = getattr(asset_cfg, "joint_names", None)
    if names is None:
        names = KNEE_JOINT_NAMES
    elif isinstance(names, str):
        names = (names,)
    else:
        names = tuple(names)
    if tuple(names) != KNEE_JOINT_NAMES:
        raise ValueError(
            "knee terms require joint_names=['Rthigh_to_Rleg', 'Lthigh_to_Lleg'] "
            f"in that order, got {tuple(names)!r}"
        )
    return tuple(names)


def _knee_bound_tensors(
    asset_cfg: SceneEntityCfg,
    device: torch.device,
    dtype: torch.dtype,
    bounds_by_joint: Mapping[str, tuple[float, float]] | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve per-side raw articulation bounds on the requested device."""

    names = _knee_joint_names(asset_cfg)
    if bounds_by_joint is None:
        if (lower is None) != (upper is None):
            raise ValueError("lower and upper must be provided together")
        if lower is not None and upper is not None:
            bounds_by_joint = {name: (float(lower), float(upper)) for name in names}
        else:
            bounds_by_joint = KNEE_ARTICULATION_LIMITS_RAD
    try:
        bounds = [bounds_by_joint[name] for name in names]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing side-specific knee bound for {names!r}") from exc
    if any(len(pair) != 2 or float(pair[0]) >= float(pair[1]) for pair in bounds):
        raise ValueError(f"knee bounds must be increasing (lower, upper), got {bounds!r}")
    lower_tensor = torch.as_tensor([float(pair[0]) for pair in bounds], device=device, dtype=dtype)
    upper_tensor = torch.as_tensor([float(pair[1]) for pair in bounds], device=device, dtype=dtype)
    return lower_tensor, upper_tensor


def knee_limit_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    bounds_by_joint: Mapping[str, tuple[float, float]] | None = None,
    lower: float | None = None,
    upper: float | None = None,
):
    """Return a positive squared cost for knee angles outside human limits.

    The manager applies a negative weight to this term.  Bounds are expressed
    in the raw articulation convention.  Right and left knees have mirrored
    axes, so a single global ``min/max`` interval would incorrectly allow a
    reverse-bent knee on one side.
    """

    joint_pos = _as_tensor(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids]
    lower_tensor, upper_tensor = _knee_bound_tensors(
        asset_cfg, joint_pos.device, joint_pos.dtype, bounds_by_joint, lower, upper
    )
    below = torch.relu(lower_tensor.unsqueeze(0) - joint_pos)
    above = torch.relu(joint_pos - upper_tensor.unsqueeze(0))
    return torch.sum(torch.square(below) + torch.square(above), dim=-1)


def knee_out_of_human_range(
    env,
    asset_cfg: SceneEntityCfg,
    bounds_by_joint: Mapping[str, tuple[float, float]] | None = None,
    lower: float | None = None,
    upper: float | None = None,
):
    """Terminate environments whose raw knee angles leave their side bounds."""

    joint_pos = _as_tensor(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids]
    lower_tensor, upper_tensor = _knee_bound_tensors(
        asset_cfg, joint_pos.device, joint_pos.dtype, bounds_by_joint, lower, upper
    )
    return torch.any(
        (joint_pos < lower_tensor.unsqueeze(0)) | (joint_pos > upper_tensor.unsqueeze(0)), dim=-1
    )


def knee_velocity_limit_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    max_velocity: float = 3.14,
):
    """Penalize knee speeds above the configured actuator safety limit."""

    if max_velocity <= 0.0:
        raise ValueError("max_velocity must be positive")
    joint_vel = _as_tensor(env.scene[asset_cfg.name].data.joint_vel)[:, asset_cfg.joint_ids]
    excess = torch.relu(torch.abs(joint_vel) - float(max_velocity))
    return torch.sum(torch.square(excess), dim=-1)


def human_knee_gait_reward(
    env,
    gait: GaitMode | int,
    asset_cfg: SceneEntityCfg,
    stance_target: float = 0.0,
    swing_target: float = 0.8,
    stance_sigma: float = 0.30,
    swing_sigma: float = 0.45,
):
    """Reward physiological stance extension and swing flexion.

    ``asset_cfg`` must contain knees in right/left order.  Contact labels are
    read in the same order and are deliberately detached from the sign of the
    raw joint angle.  This prevents PPO from receiving a positive score for a
    reverse-bent knee while still allowing flexion during the swing phase.
    """

    names = _knee_joint_names(asset_cfg)
    joint_pos = _as_tensor(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids]
    if joint_pos.shape[-1] != 2:
        raise ValueError("human_knee_gait_reward requires ordered right and left knee joints")
    # Convert raw articulation angles to anatomical flexion angles.  The
    # explicit signs are kept here rather than inferred from a tensor order.
    signs = torch.as_tensor(
        [KNEE_ARTICULATION_SIGNS[name] for name in names],
        device=joint_pos.device,
        dtype=joint_pos.dtype,
    )
    flexion = joint_pos * signs
    left_force, right_force = _foot_contacts(env)
    contact_by_joint = {
        "Rthigh_to_Rleg": right_force > CONTACT_FORCE_THRESHOLD_N,
        "Lthigh_to_Lleg": left_force > CONTACT_FORCE_THRESHOLD_N,
    }
    contact = torch.stack([contact_by_joint[name] for name in names], dim=-1)
    stance_score = torch.exp(-torch.square(flexion - float(stance_target)) / (float(stance_sigma) ** 2))
    swing_score = torch.exp(-torch.square(flexion - float(swing_target)) / (float(swing_sigma) ** 2))
    score = torch.where(contact, stance_score, swing_score).mean(dim=-1)
    # Do not shape the static stand gait with a walking knee-phase term.
    return _masked(score, env, gait)


def foot_drag_penalty(env, gait: GaitMode | int, asset_cfg: SceneEntityCfg):
    """Penalize horizontal foot speed while a foot is in contact."""

    foot_vel = _as_tensor(env.scene[asset_cfg.name].data.body_lin_vel_w)[:, asset_cfg.body_ids, :2]
    left, right = _foot_contacts(env)
    contact = torch.stack((left > CONTACT_FORCE_THRESHOLD_N, right > CONTACT_FORCE_THRESHOLD_N), dim=-1).float()
    value = torch.sum(torch.linalg.vector_norm(foot_vel, dim=-1) * contact, dim=-1)
    return _masked(value, env, gait)


def double_support_reward(env):
    """Reward simultaneous left/right contact while standing."""

    left, right = _foot_contacts(env)
    score = ((left > CONTACT_FORCE_THRESHOLD_N) & (right > CONTACT_FORCE_THRESHOLD_N)).float()
    return _masked(score, env, GaitMode.STAND)


def standing_base_stillness_reward(env, sigma: float = 0.2):
    """Reward low base velocity while standing."""

    lin, ang, _ = _base(env)
    motion = torch.sum(torch.square(lin), dim=-1) + torch.sum(torch.square(ang), dim=-1)
    return _masked(torch.exp(-motion / (sigma * sigma)), env, GaitMode.STAND)


def standing_pose_reward(env, sigma: float = 0.35):
    """Reward the default joint pose while standing."""

    data = env.scene["robot"].data
    error = _canonical_joint_tensor(env, data.joint_pos) - _canonical_joint_tensor(env, data.default_joint_pos)
    score = torch.exp(-torch.mean(torch.square(error), dim=-1) / (sigma * sigma))
    return _masked(score, env, GaitMode.STAND)


def standing_upright_reward(env, sigma: float = 0.2):
    """Reward level roll and pitch while standing."""

    roll, pitch, _ = euler_xyz_from_quat(_base(env)[2])
    score = torch.exp(-(torch.square(roll) + torch.square(pitch)) / (sigma * sigma))
    return _masked(score, env, GaitMode.STAND)


def standing_foot_flatness_reward(env, asset_cfg: SceneEntityCfg, sigma: float = 0.2):
    """Reward flat feet while standing, ignoring foot yaw."""

    foot_quat = _as_tensor(env.scene[asset_cfg.name].data.body_link_quat_w)[:, asset_cfg.body_ids]
    yaw = math_utils.yaw_quat(foot_quat.reshape(-1, 4)).reshape_as(foot_quat)
    tilt = math_utils.quat_mul(math_utils.quat_conjugate(yaw), foot_quat)
    error = torch.mean(torch.sum(torch.square(tilt[..., :3]), dim=-1), dim=-1)
    return _masked(torch.exp(-error / (sigma * sigma)), env, GaitMode.STAND)


def standing_feet_alignment_reward(env, asset_cfg: SceneEntityCfg, sigma: float = 0.2):
    """Reward both feet sharing the root yaw direction while standing."""

    foot_quat = _as_tensor(env.scene[asset_cfg.name].data.body_link_quat_w)[:, asset_cfg.body_ids]
    root_quat = _as_tensor(env.scene[asset_cfg.name].data.root_quat_w)
    desired_yaw = math_utils.yaw_quat(root_quat).unsqueeze(1).expand_as(foot_quat)
    relative = math_utils.quat_mul(math_utils.quat_conjugate(desired_yaw), math_utils.yaw_quat(foot_quat))
    error = torch.mean(torch.sum(torch.square(relative[..., :3]), dim=-1), dim=-1)
    return _masked(torch.exp(-error / (sigma * sigma)), env, GaitMode.STAND)


def standing_joint_stillness_reward(env, sigma: float = 0.25):
    """Reward low joint speed while standing."""

    joint_vel = _canonical_joint_tensor(env, env.scene["robot"].data.joint_vel)
    score = torch.exp(-torch.mean(torch.square(joint_vel), dim=-1) / (sigma * sigma))
    return _masked(score, env, GaitMode.STAND)


def smooth_slowdown_reward(env, gait: GaitMode | int, scale: float = 0.2):
    """Reward a bounded reduction in planar speed during a transition."""

    current = torch.linalg.vector_norm(_base(env)[0][:, :2], dim=-1)
    previous = _gait_command(env).previous_speed
    score = torch.clamp((previous - current) / scale, min=0.0, max=1.0)
    return _masked(score, env, gait)


def short_contact_reward(env, max_contact_time: float = 0.25):
    """Reward brief stance periods while running."""

    if not _optional_sensor_enabled(env, "contact"):
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)

    durations = []
    for sensor_name in ("left_foot_contact", "right_foot_contact"):
        duration = (
            _as_tensor(env.scene[sensor_name].data.current_contact_time).reshape(env.num_envs, -1).max(dim=-1).values
        )
        durations.append(duration)
    contact_time = torch.stack(durations, dim=-1)
    in_contact = contact_time > 0.0
    score = (torch.exp(-contact_time / max_contact_time) * in_contact.float()).sum(dim=-1)
    score /= in_contact.sum(dim=-1).clamp(min=1)
    return _masked(score, env, GaitMode.RUN)


def push_off_reward(env, asset_cfg: SceneEntityCfg):
    """Reward stance-ankle power that creates forward and upward foot motion."""

    robot = env.scene[asset_cfg.name]
    ankle_speed = torch.abs(_as_tensor(robot.data.joint_vel)[:, asset_cfg.joint_ids])
    effort = getattr(robot.data, "applied_torque", None)
    if effort is None:
        effort = getattr(robot.data, "computed_torque", None)
    if effort is None:
        effort = torch.zeros_like(_as_tensor(robot.data.joint_vel))
    ankle_effort = torch.abs(_as_tensor(effort)[:, asset_cfg.joint_ids])
    left, right = _foot_contacts(env)
    # Joint IDs are ordered [right ankle, left ankle].
    stance = torch.stack((right > CONTACT_FORCE_THRESHOLD_N, left > CONTACT_FORCE_THRESHOLD_N), dim=-1).float()
    single_support = (stance.sum(dim=-1) == 1.0).float()
    ankle_power = torch.sum(ankle_speed * ankle_effort * stance, dim=-1) / stance.sum(dim=-1).clamp(min=1.0)
    root_lin_vel = _base(env)[0]
    com_motion = torch.clamp(root_lin_vel[:, 0], min=0.0) + 0.5 * torch.clamp(root_lin_vel[:, 2], min=0.0)
    score = torch.tanh(ankle_power / 25.0) * torch.tanh(com_motion / 0.5) * single_support
    return _masked(score, env, GaitMode.RUN)


def forward_velocity_reward(env, gait: GaitMode | int, scale: float = 4.0):
    """Reward bounded positive forward velocity for a selected gait."""

    score = torch.clamp(_base(env)[0][:, 0] / scale, min=0.0, max=1.0)
    return _masked(score, env, gait)


def transition_to_walk_speed_reward(env, target_speed: float = 0.2, std: float = 0.3):
    """Reward convergence to walking speed during run-to-walk."""

    error = torch.square(_base(env)[0][:, 0] - target_speed)
    return _masked(torch.exp(-error / (std * std)), env, GaitMode.RUN_TO_WALK)


def advance_exoskeleton_curriculum(
    env,
    env_ids,
    phase2_start: int,
    phase3_start: int,
    walk_speed: float,
    transition_max_speed: float,
    multi_gait_max_speed: float,
    step_offset: int = 0,
):
    """Advance command ranges through walk, transition, and multi-gait phases.

    ``step_offset`` lets a resumed RL-Games run restore curriculum progress;
    RL-Games checkpoints do not serialize ``env.common_step_counter``.
    """

    del env_ids
    if step_offset < 0:
        raise ValueError("curriculum step_offset must be non-negative")
    curriculum_step = env.common_step_counter + step_offset
    phase = curriculum_phase_for_step(curriculum_step, phase2_start, phase3_start)
    command = _gait_command(env)
    if phase == CurriculumPhase.WALK_ONLY:
        command.cfg.ranges.lin_vel_x = (walk_speed, walk_speed)
        command.cfg.ranges.lin_vel_y = (0.0, 0.0)
        command.cfg.ranges.ang_vel_z = (0.0, 0.0)
        command.cfg.rel_standing_envs = 0.0
        max_speed = walk_speed
    elif phase == CurriculumPhase.TRANSITIONS:
        command.cfg.ranges.lin_vel_x = (walk_speed, transition_max_speed)
        command.cfg.ranges.lin_vel_y = (-0.15, 0.15)
        command.cfg.ranges.ang_vel_z = (-0.3, 0.3)
        command.cfg.rel_standing_envs = 0.2
        max_speed = transition_max_speed
    else:
        command.cfg.ranges.lin_vel_x = (walk_speed, multi_gait_max_speed)
        command.cfg.ranges.lin_vel_y = (-0.3, 0.3)
        command.cfg.ranges.ang_vel_z = (-0.5, 0.5)
        command.cfg.rel_standing_envs = 0.1
        max_speed = multi_gait_max_speed

    command.set_curriculum_phase(phase)
    return {
        "phase": float(phase),
        "control_step": float(curriculum_step),
        "max_command_speed_m_s": float(max_speed),
        "froude_threshold": FROUDE_THRESHOLD,
    }
