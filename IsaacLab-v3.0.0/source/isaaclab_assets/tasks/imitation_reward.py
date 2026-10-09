"""Multimodal motion-imitation rewards and compact sensor observations.

All similarity terms are positive scores in ``[0, 1]``.  This avoids the
sign ambiguity in the original request (which mixed similarities and errors):
the manager applies positive weights to similarities, while safety/energy
costs remain negative terms in the locomotion configuration.

The functions in this module are deliberately tensor-only at their core.  The
small Isaac Lab adapters at the bottom read the current scene and the
``MyExoskeletonImitationEnv`` target cache, so they can be used as deferred
``ObservationTermCfg``/``RewardTermCfg`` callables without importing Kit during
configuration discovery.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_assets.tasks.robot_contract import ACTUATED_JOINT_NAMES

IMU_FEATURES = 9  # roll, pitch, yaw, gyro xyz, accelerometer xyz
IMU_COUNT = 6
RGB_FEATURES = 12  # two cameras x (RGB mean + RGB standard deviation)
EVENT_FEATURES = 32  # two cameras x two polarities x eight spatial pooling bins


def _as_tensor(value: Any) -> torch.Tensor:
    """Convert Isaac Lab proxy arrays and ordinary tensors to Torch tensors."""

    return value.torch if hasattr(value, "torch") else value


def masked_exponential_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sigma: float,
    valid: torch.Tensor | None = None,
    dim: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Return ``exp(-MSE / sigma^2)`` with an optional validity mask.

    Invalid channels contribute neither error nor denominator.  Environments
    with no valid channel receive zero rather than an accidental perfect score.
    """

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    prediction, target = torch.broadcast_tensors(prediction, target)
    error = torch.square(prediction - target)
    if valid is None:
        if dim is None:
            mse = error.mean(dim=-1)
        else:
            mse = error.mean(dim=dim)
        return torch.exp(-mse / (sigma * sigma))
    mask = valid.to(dtype=error.dtype, device=error.device)
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, error.shape)
    if dim is None:
        reduce_dims = tuple(range(1, error.ndim))
    elif isinstance(dim, int):
        reduce_dims = (dim if dim >= 0 else error.ndim + dim,)
    else:
        reduce_dims = tuple(d if d >= 0 else error.ndim + d for d in dim)
    numerator = (error * mask).sum(dim=reduce_dims)
    denominator = mask.sum(dim=reduce_dims)
    score = torch.exp(-numerator / denominator.clamp_min(1.0) / (sigma * sigma))
    return torch.where(denominator > 0.0, score, torch.zeros_like(score))


def imu_orientation_similarity(
    simulation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    sigma: float = 0.35,
) -> torch.Tensor:
    """Compare the first three (Euler orientation) IMU features."""

    return masked_exponential_similarity(simulation[..., :3], target[..., :3], sigma=sigma, valid=valid, dim=(-1, -2))


def gyro_similarity(
    simulation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Compare IMU angular velocity channels."""

    return masked_exponential_similarity(simulation[..., 3:6], target[..., 3:6], sigma=sigma, valid=valid, dim=(-1, -2))


def accelerometer_similarity(
    simulation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Compare IMU linear acceleration channels."""

    return masked_exponential_similarity(simulation[..., 6:9], target[..., 6:9], sigma=sigma, valid=valid, dim=(-1, -2))


def joint_similarity(
    simulation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    sigma: float = 0.35,
) -> torch.Tensor:
    """Compare the eight canonical joint positions."""

    return masked_exponential_similarity(simulation, target, sigma=sigma, valid=valid, dim=-1)


def velocity_similarity(
    simulation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    sigma: float = 0.5,
) -> torch.Tensor:
    """Compare base linear velocity with a target velocity vector."""

    return masked_exponential_similarity(simulation, target, sigma=sigma, valid=valid, dim=-1)


def feature_similarity(
    simulation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    sigma: float = 0.25,
) -> torch.Tensor:
    """Compare compact RGB or event features."""

    return masked_exponential_similarity(simulation, target, sigma=sigma, valid=valid, dim=-1)


def weighted_imitation_reward(
    components: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    valid_components: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Combine active positive similarities using normalized weights.

    A component can be disabled by setting its weight to zero.  The denominator
    is computed per environment from active/valid channels, so missing motion or
    event files do not produce NaNs or dilute the available IMU signal.
    """

    if not components:
        raise ValueError("at least one imitation component is required")
    first = next(iter(components.values()))
    numerator = torch.zeros_like(first)
    denominator = torch.zeros_like(first)
    for name, value in components.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        active = torch.ones_like(value)
        if valid_components is not None and name in valid_components:
            active = valid_components[name].to(dtype=value.dtype, device=value.device)
        numerator = numerator + weight * value * active
        denominator = denominator + weight * active
    return torch.where(denominator > 0.0, numerator / denominator.clamp_min(1.0e-8), torch.zeros_like(numerator))


def _target_cache(env) -> dict[str, torch.Tensor]:
    cache = getattr(env, "_imitation_targets", None)
    if not isinstance(cache, dict):
        return {}
    return cache


def _target(env, name: str, shape: tuple[int, ...], *, fill: float = 0.0) -> torch.Tensor:
    value = _target_cache(env).get(name)
    if value is None:
        return torch.full((env.num_envs, *shape), fill, device=env.device, dtype=torch.float32)
    return value.to(device=env.device, dtype=torch.float32)


def _target_valid(env, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    value = _target_cache(env).get(name)
    if value is None:
        return torch.zeros((env.num_envs, *shape), device=env.device, dtype=torch.bool)
    return value.to(device=env.device, dtype=torch.bool)


def _sensor_enabled(env: Any, name: str) -> bool:
    """Read an optional sensor switch with an enabled compatibility default."""

    cfg = getattr(env, "cfg", None)
    return bool(getattr(cfg, f"use_{name}", True))


def _zero_reward(env: Any) -> torch.Tensor:
    """Return a device/dtype-correct zero reward for a disabled channel."""

    return torch.zeros((int(env.num_envs),), device=env.device, dtype=torch.float32)


def _scene_entity(env, name: str) -> Any | None:
    """Get a scene entity without relying on a dict-like ``get`` method."""

    scene = getattr(env, "scene", None)
    if scene is None:
        return None
    try:
        if name not in scene.keys():
            return None
        return scene[name]
    except (AttributeError, KeyError, TypeError):
        return None


def _robot_joint_tensor(env, field: str) -> torch.Tensor:
    robot = env.scene["robot"]
    value = _as_tensor(getattr(robot.data, field))
    names = tuple(getattr(robot, "joint_names", ()))
    ids = getattr(env, "_imitation_joint_ids", None)
    if ids is None:
        missing = [name for name in ACTUATED_JOINT_NAMES if name not in names]
        if missing:
            raise ValueError(f"robot is missing canonical joints: {missing}")
        ids = torch.tensor([names.index(name) for name in ACTUATED_JOINT_NAMES], device=value.device)
        setattr(env, "_imitation_joint_ids", ids)
    return value.index_select(-1, ids)


def _sensor_tensor(sensor: Any, field: str, env_count: int, device: torch.device | str) -> torch.Tensor:
    if sensor is None:
        return torch.zeros((env_count, 3), device=device)
    data = sensor.data
    value = getattr(data, field, None)
    if value is None:
        return torch.zeros((env_count, 3), device=device)
    value = _as_tensor(value).to(device=device, dtype=torch.float32)
    if value.ndim == 1:
        value = value.reshape(env_count, -1)
    return value.reshape(env_count, -1)[..., :3]


def _imu_sensor_names(env) -> tuple[str, ...]:
    return tuple(
        getattr(env, "imitation_imu_sensor_names", ("imu_01", "imu_02", "imu_03", "imu_04", "imu_05", "imu_06"))
    )


def _body_orientation(env, body_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    quat = _as_tensor(robot.data.root_quat_w)
    names = tuple(getattr(robot, "body_names", ()))
    if body_name in names and hasattr(robot.data, "body_link_quat_w"):
        quat = _as_tensor(robot.data.body_link_quat_w)[:, names.index(body_name)]
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    return torch.stack((roll, pitch, yaw), dim=-1)


def imu_observation(env, *, inspect: bool = False) -> torch.Tensor:
    """Return ``obs["imu"]`` as a flattened ``[N, 6*9]`` feature tensor."""

    if inspect or not _sensor_enabled(env, "imu"):
        env._latest_imu = torch.zeros((env.num_envs, IMU_COUNT, IMU_FEATURES), device=env.device)
        env._latest_imu_valid = torch.zeros(
            (env.num_envs, IMU_COUNT), dtype=torch.bool, device=env.device
        )
        return torch.zeros((env.num_envs, IMU_COUNT * IMU_FEATURES), device=env.device)
    body_names = tuple(
        getattr(env, "_resolved_imu_links", ("Lfoot", "Rfoot", "Lleg", "Rleg", "Lthigh", "Rthigh"))
    )
    rows = []
    for sensor_name, body_name in zip(_imu_sensor_names(env), body_names, strict=True):
        orientation = _body_orientation(env, body_name)
        sensor = _scene_entity(env, sensor_name)
        gyro = _sensor_tensor(sensor, "ang_vel_b", env.num_envs, env.device)
        accel = _sensor_tensor(sensor, "lin_acc_b", env.num_envs, env.device)
        rows.append(torch.cat((orientation, gyro, accel), dim=-1))
    stacked = torch.stack(rows, dim=1)
    # Keep the native [N, 6, 9] tensor for play/viser diagnostics while the
    # manager-facing term remains a flat vector for rl_games concatenation.
    env._latest_imu = stacked
    env._latest_imu_valid = torch.stack(
        [
            torch.full(
                (env.num_envs,),
                _scene_entity(env, sensor_name) is not None,
                dtype=torch.bool,
                device=env.device,
            )
            for sensor_name in _imu_sensor_names(env)
        ],
        dim=1,
    )
    return stacked.reshape(env.num_envs, -1)


def camera_rgb_observation(env, *, inspect: bool = False) -> torch.Tensor:
    """Return compact RGB moments while retaining raw frames for playback."""

    if inspect or not _sensor_enabled(env, "rgb"):
        env._latest_rgb_valid = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        env._latest_rgb_feature = torch.zeros((env.num_envs, RGB_FEATURES), device=env.device)
        env._latest_camera_rgb = {}
        return torch.zeros((env.num_envs, RGB_FEATURES), device=env.device)
    means = []
    stds = []
    raw = {}
    available = torch.ones((env.num_envs,), dtype=torch.bool, device=env.device)
    for name in ("left_rgb_camera", "right_rgb_camera"):
        camera = _scene_entity(env, name)
        image = None
        if camera is not None:
            try:
                image = _as_tensor(camera.data.output["rgb"]).to(device=env.device, dtype=torch.float32)
            except (KeyError, AttributeError, RuntimeError):
                image = None
        # RTX camera buffers can be empty during ObservationManager's initial
        # shape probe (before the first render).  Treat that state as an
        # unavailable frame instead of reducing an invalid CUDA view.
        if (
            image is None
            or image.ndim != 4
            or image.shape[0] != env.num_envs
            or image.shape[1] < 1
            or image.shape[2] < 1
            or image.shape[3] < 3
        ):
            available.zero_()
            image = torch.zeros((env.num_envs, 256, 256, 3), device=env.device)
        elif image.shape[-1] > 3:
            image = image[..., :3]
        # Isaac Lab's renderer returns uint8-range RGB.  The dataset loader
        # and feature reward use [0, 1] floats, so normalize exactly once.
        try:
            if image.numel() and float(image.detach().amax()) > 1.5:
                image = image / 255.0
            means.append(image.mean(dim=(1, 2)))
            stds.append(image.std(dim=(1, 2), unbiased=False))
        except (RuntimeError, TypeError, ValueError):
            # A renderer can invalidate its buffer while a headless stage is
            # being reset.  Keep the observation shape stable for PPO.
            available.zero_()
            image = torch.zeros((env.num_envs, 256, 256, 3), device=env.device)
            means.append(image.mean(dim=(1, 2)))
            stds.append(image.std(dim=(1, 2), unbiased=False))
        if bool(getattr(env.cfg, "store_raw_camera_frames", False)) or env.num_envs <= 4:
            raw[name] = image
    feature = torch.cat((*means, *stds), dim=-1)
    env._latest_rgb_feature = feature
    env._latest_camera_rgb = raw
    env._latest_rgb_valid = available
    return feature


def event_camera_observation(env, *, inspect: bool = False) -> torch.Tensor:
    """Return fixed event histograms from EVIS, or invalid zeros without it."""

    if inspect or not _sensor_enabled(env, "event"):
        env._latest_event_hist = torch.zeros((env.num_envs, EVENT_FEATURES), device=env.device)
        env._latest_event_valid = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        return torch.zeros((env.num_envs, EVENT_FEATURES), device=env.device)
    update = getattr(env, "update_event_observation", None)
    if callable(update):
        value, valid = update()
        env._latest_event_valid = valid
        return value
    env._latest_event_valid = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    return torch.zeros((env.num_envs, EVENT_FEATURES), device=env.device)


def imitation_imu_reward(env, orientation_sigma: float = 0.35, gyro_sigma: float = 1.0, accel_sigma: float = 2.0):
    if not _sensor_enabled(env, "imu"):
        return _zero_reward(env)
    simulation = imu_observation(env)
    simulation = simulation.reshape(env.num_envs, IMU_COUNT, IMU_FEATURES)
    target = _target(env, "target_imu", (IMU_COUNT, IMU_FEATURES))
    valid = _target_valid(env, "imu_valid", (IMU_COUNT,))
    pose = imu_orientation_similarity(simulation, target, valid, orientation_sigma)
    gyro = gyro_similarity(simulation, target, valid, gyro_sigma)
    accel = accelerometer_similarity(simulation, target, valid, accel_sigma)
    return (pose + 0.5 * gyro + 0.25 * accel) / 1.75


def imitation_gyro_reward(env, sigma: float = 1.0):
    if not _sensor_enabled(env, "imu"):
        return _zero_reward(env)
    simulation = imu_observation(env).reshape(env.num_envs, IMU_COUNT, IMU_FEATURES)
    target = _target(env, "target_imu", (IMU_COUNT, IMU_FEATURES))
    valid = _target_valid(env, "imu_valid", (IMU_COUNT,))
    return gyro_similarity(simulation, target, valid, sigma)


def imitation_joint_reward(env, sigma: float = 0.35):
    simulation = _robot_joint_tensor(env, "joint_pos")
    target = _target(env, "target_motion", (len(ACTUATED_JOINT_NAMES),))
    valid = _target_valid(env, "motion_valid", ())
    return joint_similarity(simulation, target, valid, sigma)


def imitation_velocity_reward(env, sigma: float = 0.5):
    simulation = _as_tensor(env.scene["robot"].data.root_lin_vel_b)[:, :2]
    target = _target(env, "target_velocity", (2,))
    valid = _target_valid(env, "velocity_valid", ())
    return velocity_similarity(simulation, target, valid, sigma)


def imitation_rgb_reward(env, sigma: float = 0.25):
    if not _sensor_enabled(env, "rgb"):
        return _zero_reward(env)
    simulation = camera_rgb_observation(env)
    target = _target(env, "target_rgb_feature", (RGB_FEATURES,))
    valid = _target_valid(env, "rgb_valid", ()) & getattr(
        env, "_latest_rgb_valid", torch.ones((env.num_envs,), dtype=torch.bool, device=env.device)
    )
    return feature_similarity(simulation, target, valid, sigma)


def imitation_event_reward(env, sigma: float = 1.0):
    if not _sensor_enabled(env, "event"):
        return _zero_reward(env)
    simulation = event_camera_observation(env)
    target = _target(env, "target_event_feature", (EVENT_FEATURES,))
    target_valid = _target_valid(env, "event_valid", ())
    runtime_valid = getattr(
        env,
        "_latest_event_valid",
        torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
    )
    valid = target_valid & runtime_valid.to(device=env.device, dtype=torch.bool)
    return feature_similarity(simulation, target, valid, sigma)


def imitation_total_reward(
    env,
    weights: Mapping[str, float] | None = None,
    orientation_sigma: float = 0.35,
    gyro_sigma: float = 1.0,
    joint_sigma: float = 0.35,
    velocity_sigma: float = 0.5,
    rgb_sigma: float = 0.25,
    event_sigma: float = 1.0,
):
    """Compute the normalized six-channel multimodal imitation reward."""

    weights = weights or {
        "imu": 1.0,
        "gyro": 0.5,
        "joint": 0.5,
        "velocity": 0.5,
        "rgb": 0.25,
        "event": 0.25,
    }
    components = {
        "imu": imitation_imu_reward(env, orientation_sigma, gyro_sigma),
        "gyro": imitation_gyro_reward(env, gyro_sigma),
        "joint": imitation_joint_reward(env, joint_sigma),
        "velocity": imitation_velocity_reward(env, velocity_sigma),
        "rgb": imitation_rgb_reward(env, rgb_sigma),
        "event": imitation_event_reward(env, event_sigma),
    }
    valid = {
        "imu": _sensor_enabled(env, "imu") & _target_valid(env, "imu_valid", (IMU_COUNT,)).any(dim=-1),
        "gyro": _sensor_enabled(env, "imu") & _target_valid(env, "imu_valid", (IMU_COUNT,)).any(dim=-1),
        "joint": _target_valid(env, "motion_valid", ()).reshape(env.num_envs),
        "velocity": _target_valid(env, "velocity_valid", ()).reshape(env.num_envs),
        "rgb": (
            _sensor_enabled(env, "rgb")
            & _target_valid(env, "rgb_valid", ()).reshape(env.num_envs)
            & getattr(
                env,
                "_latest_rgb_valid",
                torch.ones((env.num_envs,), dtype=torch.bool, device=env.device),
            )
        ),
        "event": (
            _sensor_enabled(env, "event")
            & _target_valid(env, "event_valid", ()).reshape(env.num_envs)
            & getattr(
                env,
                "_latest_event_valid",
                torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
            ).to(device=env.device, dtype=torch.bool)
        ),
    }
    return weighted_imitation_reward(components, weights, valid)


__all__ = [
    "EVENT_FEATURES",
    "IMU_COUNT",
    "IMU_FEATURES",
    "RGB_FEATURES",
    "accelerometer_similarity",
    "camera_rgb_observation",
    "event_camera_observation",
    "feature_similarity",
    "gyro_similarity",
    "imu_observation",
    "imitation_event_reward",
    "imitation_gyro_reward",
    "imitation_imu_reward",
    "imitation_joint_reward",
    "imitation_rgb_reward",
    "imitation_total_reward",
    "imitation_velocity_reward",
    "joint_similarity",
    "masked_exponential_similarity",
    "velocity_similarity",
    "weighted_imitation_reward",
]
