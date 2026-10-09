# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""把策略回放转换为结构化关节轨迹输出。

重要前提：当前 PPO actor 仍只输出 8 个归一化动作。本模块在环境完成一步之后读取 8 个
处理后的目标角度，再结合相邻采样时间计算目标速度和角加速度，形成 24 个物理量；它
没有把旧 checkpoint 的网络输出层偷偷改成 24 维。

RuleresultRecorderMixin 被基础环境和模仿环境共同继承。每次 env.step 后，它按固定
公共关节顺序生成记录，可写为可读 JSONL，或经过标定与安全检查后写成 55AA 二进制
帧。measured 字段是仿真实际反馈，target 字段才是策略目标，二者不能混用。

时间模式可以使用仿真时间、墙钟时间或固定配置间隔。非固定间隔差分必须使用相邻记录
的真实 dt；reset 时必须清空历史，避免把新 episode 第一帧减去旧 episode 末帧。
未来 kinematic 序列只是运动学外推，supplied 才表示外部真正提供的未来目标。
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from isaaclab_assets.tasks.protocol_55aa import Protocol55AAConfig
from isaaclab_assets.tasks.robot_contract import ACTUATED_JOINT_NAMES, CONTROL_DT_S

RULERESULT_SCHEMA = "ruleresult/v1"
# 结果按字段分组排列：8 个角度 + 8 个速度 + 8 个角加速度 = 24，而不是每个关节
# 连续放 q/v/a。JSONL 还会保存字段名，因此比裸 24 维张量更适合人工检查。
RULERESULT_DIM = 3 * len(ACTUATED_JOINT_NAMES)
RULERESULT_OUTPUT_FORMATS = ("jsonl", "protocol_55aa")
RULERESULT_TIME_MODES = ("simulation", "wall_clock", "configured")
RULERESULT_PREDICTION_MODES = ("none", "kinematic", "supplied")


def _as_tensor(value: Any) -> torch.Tensor:
    """把普通张量或 Isaac Lab 代理数组统一转换为 Torch 张量视图。"""

    value = value.torch if hasattr(value, "torch") else value
    return torch.as_tensor(value)


def _reorder_joint_tensor(value: torch.Tensor, names: Sequence[str]) -> torch.Tensor:
    """按公共 8 关节约定重排张量最后一维。

    输入可以带任意前导批量维，例如 ``[N, 8]`` 或 ``[N, H, 8]``；只有最后的关节维
    会重排。这样即使 USD、动作项和导出协议使用不同内部顺序，公开结果仍能稳定按
    :data:`ACTUATED_JOINT_NAMES` 解释。
    """

    if value.ndim < 1 or value.shape[-1] != len(names):
        raise ValueError(f"expected a final dimension of {len(names)}, got {tuple(value.shape)}")
    if tuple(names) == ACTUATED_JOINT_NAMES:
        return value
    source_names = tuple(names)
    missing = [name for name in ACTUATED_JOINT_NAMES if name not in source_names]
    if missing:
        raise ValueError(f"joint tensor is missing canonical joints: {missing}")
    indices = torch.tensor(
        [source_names.index(name) for name in ACTUATED_JOINT_NAMES], device=value.device, dtype=torch.long
    )
    return value.index_select(-1, indices)


def derive_target_kinematics(
    angle_target_rad: torch.Tensor,
    previous_angle_target_rad: torch.Tensor | None = None,
    previous_velocity_rad_s: torch.Tensor | None = None,
    dt_s: float = CONTROL_DT_S,
    previous_dt_s: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """根据相邻目标角和实际采样间隔计算目标速度、目标角加速度。

    使用后向差分 ``v[t]=(q[t]-q[t-1])/dt[t]``。当本次与上次间隔不同，角加速度
    使用两段间隔的中点 ``0.5*(dt[t]+dt[t-1])``，避免把运行卡顿或调度抖动错误地
    当成固定 20 ms。该函数计算的是“目标轨迹导数”，不是电机反馈导数。

    Args:
        angle_target_rad: 当前目标角 [rad]，形状 ``[..., 8]``。
        previous_angle_target_rad: 上一控制步目标角；``None`` 表示没有可用历史。
        previous_velocity_rad_s: 上一步目标速度 [rad/s]；``None`` 表示不能计算加速度。
        dt_s: 从上一次样本到当前样本的间隔 [s]。
        previous_dt_s: 再前一段采样间隔 [s]；提供时用于非均匀加速度差分。

    Returns:
        ``(目标速度 [rad/s], 目标角加速度 [rad/s^2])``，形状与输入相同。reset 后
        第一帧没有前一帧可比较，两种导数明确返回零，不能把旧 episode 末帧当历史。
    """

    if not math.isfinite(float(dt_s)) or dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive and finite, got {dt_s}")
    if previous_dt_s is not None and (not math.isfinite(float(previous_dt_s)) or previous_dt_s <= 0.0):
        raise ValueError(f"previous_dt_s must be positive and finite, got {previous_dt_s}")
    angle = _as_tensor(angle_target_rad).detach()
    if not (angle.is_floating_point() or angle.is_complex()):
        angle = angle.float()
    if angle.is_complex():
        raise ValueError("angle_target_rad must be real-valued")
    if angle.shape[-1] != len(ACTUATED_JOINT_NAMES):
        raise ValueError(f"expected {len(ACTUATED_JOINT_NAMES)} joint targets, got {tuple(angle.shape)}")

    has_previous_angle = previous_angle_target_rad is not None
    if not has_previous_angle:
        velocity = torch.zeros_like(angle)
    else:
        previous_angle = _as_tensor(previous_angle_target_rad).to(device=angle.device, dtype=angle.dtype)
        if previous_angle.shape != angle.shape:
            raise ValueError(f"previous angle shape {tuple(previous_angle.shape)} != {tuple(angle.shape)}")
        velocity = (angle - previous_angle) / dt_s

    # reset 后第一帧没有有意义的加速度；即便调用方误传了旧速度，也明确输出零。
    if not has_previous_angle or previous_velocity_rad_s is None:
        acceleration = torch.zeros_like(angle)
    else:
        previous_velocity = _as_tensor(previous_velocity_rad_s).to(device=angle.device, dtype=angle.dtype)
        if previous_velocity.shape != angle.shape:
            raise ValueError(f"previous velocity shape {tuple(previous_velocity.shape)} != {tuple(angle.shape)}")
        acceleration_dt = dt_s if previous_dt_s is None else 0.5 * (dt_s + previous_dt_s)
        acceleration = (velocity - previous_velocity) / acceleration_dt
    return velocity, acceleration


def _prediction_timestamps(
    *,
    batch_size: int,
    horizon: int,
    timestamps_s: torch.Tensor | None,
    dt_s: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """把未来相对时间统一为形状 ``[N, H]`` 的严格递增张量 [s]。

    ``N`` 是并行环境数，``H`` 是未来步数。调用者既可以给所有环境共用的 ``[H]``，
    也可以给每个环境不同的 ``[N, H]``；未提供时按固定 ``dt_s`` 生成。
    """

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not math.isfinite(float(dt_s)) or dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive and finite, got {dt_s}")
    if timestamps_s is None:
        base = torch.arange(1, horizon + 1, device=device, dtype=dtype) * float(dt_s)
        return base.unsqueeze(0).expand(batch_size, -1)

    timestamps = _as_tensor(timestamps_s).to(device=device, dtype=dtype)
    if timestamps.ndim == 1:
        if timestamps.shape[0] != horizon:
            raise ValueError(f"timestamps_s must have {horizon} entries, got {tuple(timestamps.shape)}")
        timestamps = timestamps.unsqueeze(0).expand(batch_size, -1)
    elif timestamps.ndim == 2:
        if tuple(timestamps.shape) != (batch_size, horizon):
            raise ValueError(f"timestamps_s must have shape {(batch_size, horizon)}, got {tuple(timestamps.shape)}")
    else:
        raise ValueError("timestamps_s must have shape [H] or [N, H]")
    if not bool(torch.isfinite(timestamps).all()):
        raise ValueError("timestamps_s must contain only finite values")
    deltas = timestamps[:, 1:] - timestamps[:, :-1]
    if deltas.numel() and not bool((deltas > 0.0).all()):
        raise ValueError("timestamps_s must be strictly increasing")
    first_delta = timestamps[:, 0]
    if not bool((first_delta > 0.0).all()):
        raise ValueError("the first prediction timestamp must be positive")
    return timestamps


def predict_ruleresult_sequence(
    angle_targets_rad: torch.Tensor,
    timestamps_s: torch.Tensor | None = None,
    dt_s: float = CONTROL_DT_S,
    initial_velocity_rad_s: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """按非均匀未来时间戳对一段已提供的目标角序列求导。

    本函数不会调用策略，也不会向前运行仿真。它只接收由外部规划器、策略 rollout 或
    数据集已经给出的未来角度，再按每段真实时间差计算速度和加速度。因此结果可称为
    “已提供序列的导数”，不能宣称 PPO 自己预测了未来。输入为 ``[H, 8]`` 或
    ``[N, H, 8]``，输出保持同样是否带批量维的形式。

    Args:
        angle_targets_rad: 未来位置目标 [rad]，形状 ``[H, 8]`` 或 ``[N, H, 8]``。
        timestamps_s: 相对当前时刻的未来时间 [s]，形状 ``[H]`` 或 ``[N, H]``。
            省略时用 ``dt_s`` 生成均匀时间轴。
        dt_s: 未提供时间戳时使用的固定未来间隔 [s]。
        initial_velocity_rad_s: 当前边界速度 [rad/s]，形状 ``[8]`` 或 ``[N, 8]``。

    Returns:
        含 ``angles``、``velocities``、``accelerations``、``timestamps_s``、
        ``dt_s`` 和 ``valid_mask`` 的字典。第一未来点之前没有另一未来目标，所以其
        差分有效标记为假；若给出初始速度，它只作为求后续加速度的边界条件。
    """

    angle = _as_tensor(angle_targets_rad).detach()
    if not (angle.is_floating_point() or angle.is_complex()):
        angle = angle.float()
    if angle.is_complex():
        raise ValueError("angle_targets_rad must be real-valued")
    if angle.ndim not in (2, 3) or angle.shape[-1] != len(ACTUATED_JOINT_NAMES):
        raise ValueError(f"angle_targets_rad must have shape [H, {len(ACTUATED_JOINT_NAMES)}] or [N, H, 8]")
    was_unbatched = angle.ndim == 2
    if was_unbatched:
        angle = angle.unsqueeze(0)
    batch_size, horizon, joint_count = angle.shape
    if horizon <= 0:
        raise ValueError("angle_targets_rad must contain at least one future sample")
    if not bool(torch.isfinite(angle).all()):
        raise ValueError("angle_targets_rad must contain only finite values")

    timestamps = _prediction_timestamps(
        batch_size=batch_size,
        horizon=horizon,
        timestamps_s=timestamps_s,
        dt_s=dt_s,
        device=angle.device,
        dtype=angle.dtype,
    )
    deltas = torch.empty_like(timestamps)
    deltas[:, 0] = timestamps[:, 0]
    if horizon > 1:
        deltas[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]

    velocities = torch.zeros_like(angle)
    if initial_velocity_rad_s is not None:
        initial_velocity = _as_tensor(initial_velocity_rad_s).to(device=angle.device, dtype=angle.dtype)
        if initial_velocity.ndim == 1:
            if initial_velocity.shape != (joint_count,):
                raise ValueError(
                    f"initial_velocity_rad_s must have shape {(joint_count,)}, got {tuple(initial_velocity.shape)}"
                )
            initial_velocity = initial_velocity.unsqueeze(0).expand(batch_size, -1)
        elif tuple(initial_velocity.shape) != (batch_size, joint_count):
            raise ValueError(
                f"initial_velocity_rad_s must have shape {(batch_size, joint_count)}, "
                f"got {tuple(initial_velocity.shape)}"
            )
        if not bool(torch.isfinite(initial_velocity).all()):
            raise ValueError("initial_velocity_rad_s must contain only finite values")
        velocities[:, 0] = initial_velocity
    if horizon > 1:
        velocities[:, 1:] = (angle[:, 1:] - angle[:, :-1]) / deltas[:, 1:, None]

    accelerations = torch.zeros_like(angle)
    if horizon > 1:
        acceleration_deltas = torch.empty_like(deltas[:, 1:])
        acceleration_deltas[:, 0] = 0.5 * (deltas[:, 0] + deltas[:, 1])
        if horizon > 2:
            acceleration_deltas[:, 1:] = 0.5 * (deltas[:, 1:-1] + deltas[:, 2:])
        accelerations[:, 1:] = (velocities[:, 1:] - velocities[:, :-1]) / acceleration_deltas[:, :, None]
    valid_mask = torch.zeros((batch_size, horizon), dtype=torch.bool, device=angle.device)
    if horizon > 1:
        valid_mask[:, 1:] = True

    result = {
        "angles": angle,
        "velocities": velocities,
        "accelerations": accelerations,
        "timestamps_s": timestamps,
        "dt_s": deltas,
        "valid_mask": valid_mask,
    }
    if was_unbatched:
        return {key: value.squeeze(0) for key, value in result.items()}
    return result


def extrapolate_ruleresult_sequence(
    angle_rad: torch.Tensor,
    velocity_rad_s: torch.Tensor,
    angular_acceleration_rad_s2: torch.Tensor,
    horizon_steps: int,
    dt_s: float = CONTROL_DT_S,
) -> dict[str, torch.Tensor]:
    """用当前状态生成带明确标签的恒加速度未来轨迹。

    公式是纯运动学 ``q=q0+v0*t+0.5*a*t^2``、``v=v0+a*t``。它只是把最新目标状态
    向未来外推，不读取未来观测，也不等价于 PPO actor 连续推理。因此 JSON 中会标为
    ``constant_acceleration_extrapolation``。输入形状为 ``[8]`` 或 ``[N, 8]``，输出
    形状为 ``[H, 8]`` 或 ``[N, H, 8]``，单位分别为 rad、rad/s、rad/s^2。
    """

    if horizon_steps <= 0:
        raise ValueError(f"horizon_steps must be positive, got {horizon_steps}")
    if not math.isfinite(float(dt_s)) or dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive and finite, got {dt_s}")
    angle = _as_tensor(angle_rad).detach()
    if not (angle.is_floating_point() or angle.is_complex()):
        angle = angle.float()
    if angle.is_complex():
        raise ValueError("angle_rad must be real-valued")
    velocity = _as_tensor(velocity_rad_s).detach().to(device=angle.device, dtype=angle.dtype)
    acceleration = _as_tensor(angular_acceleration_rad_s2).detach().to(device=angle.device, dtype=angle.dtype)
    if angle.shape != velocity.shape or angle.shape != acceleration.shape:
        raise ValueError("angle, velocity, and acceleration must have matching shapes")
    if angle.ndim not in (1, 2) or angle.shape[-1] != len(ACTUATED_JOINT_NAMES):
        raise ValueError("state tensors must have shape [8] or [N, 8]")
    if not bool(torch.isfinite(angle).all() and torch.isfinite(velocity).all() and torch.isfinite(acceleration).all()):
        raise ValueError("state tensors must contain only finite values")
    unbatched = angle.ndim == 1
    if unbatched:
        angle, velocity, acceleration = (value.unsqueeze(0) for value in (angle, velocity, acceleration))
    times = torch.arange(1, horizon_steps + 1, device=angle.device, dtype=angle.dtype) * float(dt_s)
    times_batch = times.unsqueeze(0).expand(angle.shape[0], -1)
    future_angle = angle[:, None, :] + velocity[:, None, :] * times_batch[:, :, None]
    future_angle = future_angle + 0.5 * acceleration[:, None, :] * times_batch[:, :, None].square()
    future_velocity = velocity[:, None, :] + acceleration[:, None, :] * times_batch[:, :, None]
    future_acceleration = acceleration[:, None, :].expand(-1, horizon_steps, -1).clone()
    result = {
        "angles": future_angle,
        "velocities": future_velocity,
        "accelerations": future_acceleration,
        "timestamps_s": times_batch,
        "dt_s": torch.full_like(times_batch, float(dt_s)),
        "valid_mask": torch.ones((angle.shape[0], horizon_steps), dtype=torch.bool, device=angle.device),
    }
    if unbatched:
        return {key: value.squeeze(0) for key, value in result.items()}
    return result


def flatten_ruleresult(
    angle_rad: torch.Tensor, velocity_rad_s: torch.Tensor, angular_acceleration_rad_s2: torch.Tensor
) -> torch.Tensor:
    """按字段拼成 ``[..., 24]``：先8角度，再8速度，最后8角加速度。"""

    tensors = tuple(_as_tensor(value) for value in (angle_rad, velocity_rad_s, angular_acceleration_rad_s2))
    expected = tensors[0].shape
    if expected[-1] != len(ACTUATED_JOINT_NAMES) or any(value.shape != expected for value in tensors[1:]):
        raise ValueError("angle, velocity, and acceleration must have matching [..., 8] shapes")
    return torch.cat(tensors, dim=-1)


def _vector_list(value: torch.Tensor) -> list[float]:
    """把一个有限 ``[8]`` 关节向量转换为可写入 JSON 的 Python 浮点列表。"""

    value = _as_tensor(value).detach().float().reshape(-1)
    if value.numel() != len(ACTUATED_JOINT_NAMES):
        raise ValueError(f"expected {len(ACTUATED_JOINT_NAMES)} values, got {value.numel()}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("ruleresult contains a non-finite joint value")
    return [float(item) for item in value.cpu().tolist()]


def _sequence_lists(value: torch.Tensor, *, joint_count: int = len(ACTUATED_JOINT_NAMES)) -> list[list[float]]:
    """把 ``[H, 8]`` 未来序列转换为 JSON 可序列化的二维列表。"""

    tensor = _as_tensor(value).detach().float()
    if tensor.ndim != 2 or tensor.shape[-1] != joint_count:
        raise ValueError(f"expected a [H, {joint_count}] sequence, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("ruleresult prediction contains a non-finite value")
    return [[float(item) for item in row] for row in tensor.cpu().tolist()]


def _prediction_record(prediction: dict[str, Any]) -> dict[str, Any]:
    """校验一段未来序列，并转换成 JSONL 中统一的 prediction 对象。

    该层要求角度、速度、加速度均为 ``[H, 8]``，相对时间为严格递增的 ``[H]``。
    ``source`` 会保留未来序列来源，使读取者能区分恒加速度外推和外部真实提供的目标。
    """

    required = ("angles", "velocities", "accelerations")
    missing = [key for key in required if key not in prediction]
    if missing:
        raise ValueError(f"prediction is missing fields: {missing}")
    angles = _as_tensor(prediction["angles"])
    velocities = _as_tensor(prediction["velocities"])
    accelerations = _as_tensor(prediction["accelerations"])
    if angles.ndim != 2 or angles.shape[-1] != len(ACTUATED_JOINT_NAMES):
        raise ValueError("prediction angles must have shape [H, 8]")
    if velocities.shape != angles.shape or accelerations.shape != angles.shape:
        raise ValueError("prediction angle, velocity, and acceleration shapes must match")
    horizon = int(angles.shape[0])
    if horizon <= 0:
        raise ValueError("prediction horizon must be positive")
    timestamps = prediction.get("timestamps_s")
    if timestamps is None:
        raise ValueError("prediction must include timestamps_s")
    timestamp_values = _as_tensor(timestamps).detach().float().reshape(-1)
    if timestamp_values.numel() != horizon or not bool(torch.isfinite(timestamp_values).all()):
        raise ValueError("prediction timestamps_s must contain one finite value per horizon step")
    if not bool((timestamp_values > 0.0).all()):
        raise ValueError("prediction timestamps_s must be positive relative times")
    if horizon > 1 and not bool(torch.all(torch.diff(timestamp_values) > 0.0)):
        raise ValueError("prediction timestamps_s must be strictly increasing")
    valid_mask = prediction.get("valid_mask")
    if valid_mask is None:
        valid_values = [True] * horizon
    else:
        valid_tensor = _as_tensor(valid_mask).detach().bool().reshape(-1)
        if valid_tensor.numel() != horizon:
            raise ValueError("prediction valid_mask must contain one value per horizon step")
        valid_values = [bool(item) for item in valid_tensor.cpu().tolist()]
    result: dict[str, Any] = {
        "source": str(prediction.get("source", "supplied")),
        "horizon_steps": horizon,
        "timestamps_s": [float(item) for item in timestamp_values.cpu().tolist()],
        "angle_rad": _sequence_lists(angles),
        "velocity_rad_s": _sequence_lists(velocities),
        "angular_acceleration_rad_s2": _sequence_lists(accelerations),
        "valid_mask": valid_values,
    }
    prediction_dt = prediction.get("dt_s")
    if prediction_dt is not None:
        dt_values = _as_tensor(prediction_dt).detach().float().reshape(-1)
        if (
            dt_values.numel() not in (1, horizon)
            or not bool(torch.isfinite(dt_values).all())
            or not bool((dt_values > 0.0).all())
        ):
            raise ValueError("prediction dt_s must be positive and finite")
        result["dt_s"] = [float(item) for item in dt_values.cpu().tolist()]
    return result


def build_ruleresult_record(
    *,
    step: int,
    timestamp_s: float,
    angle_rad: torch.Tensor,
    velocity_rad_s: torch.Tensor,
    angular_acceleration_rad_s2: torch.Tensor,
    env_id: int = 0,
    measured_angle_rad: torch.Tensor | None = None,
    measured_velocity_rad_s: torch.Tensor | None = None,
    measured_angular_acceleration_rad_s2: torch.Tensor | None = None,
    raw_action: torch.Tensor | None = None,
    episode: int = 0,
    dt_s: float = CONTROL_DT_S,
    control_dt_s: float | None = None,
    output_dt_s: float | None = None,
    dt_source: str = "configured",
    derivative_valid: bool = True,
    time_gap_detected: bool = False,
    wall_timestamp_s: float | None = None,
    wall_dt_s: float | None = None,
    prediction: dict[str, Any] | None = None,
    prediction_status: str | None = None,
    derivative_history_valid: bool = True,
) -> dict[str, Any]:
    """构造一行可直接写入 JSONL、且带单位和上下文的结果记录。

    ``angle_rad``、``velocity_rad_s``、``angular_acceleration_rad_s2`` 是策略位置目标及
    其导数；``measured_*`` 是仿真关节实际反馈。二者并列保存便于分析跟踪误差，但
    55AA 输出只编码目标字段。``sample_dt_s`` 是本次差分实际采用的间隔，
    ``control_dt_s`` 是标称策略周期，``output_dt_s`` 则是相邻落盘记录间隔；当
    ``ruleresult_interval > 1`` 时三者不会相等。

    每个关节向量形状为 ``[8]``，单位写在字段名中。``prediction`` 可选，不提供未来
    目标时当前样本仍是完整有效的 ruleresult。
    """

    if step < 0 or env_id < 0 or episode < 0:
        raise ValueError("step, env_id, and episode must be non-negative")
    if not math.isfinite(float(timestamp_s)):
        raise ValueError("timestamp_s must be finite")
    if not math.isfinite(float(dt_s)) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if control_dt_s is None:
        control_dt_s = dt_s
    if not math.isfinite(float(control_dt_s)) or control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be finite and positive")
    for name, value in (("output_dt_s", output_dt_s), ("wall_timestamp_s", wall_timestamp_s), ("wall_dt_s", wall_dt_s)):
        if value is not None and (not math.isfinite(float(value)) or (name.endswith("dt_s") and value <= 0.0)):
            raise ValueError(f"{name} must be finite" + (" and positive" if name.endswith("dt_s") else ""))
    if not isinstance(dt_source, str) or not dt_source:
        raise ValueError("dt_source must be a non-empty string")
    record: dict[str, Any] = {
        "schema": RULERESULT_SCHEMA,
        "step": int(step),
        "episode": int(episode),
        "env_id": int(env_id),
        "timestamp_s": float(timestamp_s),
        # ``dt_s`` 是为兼容 v1 保留的别名，表示导数计算使用的采样间隔。
        # 新的读取程序应改用 ``sample_dt_s``。
        "dt_s": float(dt_s),
        "control_dt_s": float(control_dt_s),
        "sample_dt_s": float(dt_s),
        "dt_source": dt_source,
        "derivative_valid": bool(derivative_valid),
        "derivative_history_valid": bool(derivative_history_valid),
        "time_gap_detected": bool(time_gap_detected),
        "joint_names": list(ACTUATED_JOINT_NAMES),
        "angle_rad": _vector_list(angle_rad),
        "velocity_rad_s": _vector_list(velocity_rad_s),
        "angular_acceleration_rad_s2": _vector_list(angular_acceleration_rad_s2),
    }
    if raw_action is not None:
        record["raw_action"] = _vector_list(raw_action)
    if output_dt_s is not None:
        record["output_dt_s"] = float(output_dt_s)
    if wall_timestamp_s is not None:
        record["wall_timestamp_s"] = float(wall_timestamp_s)
    if wall_dt_s is not None:
        record["wall_dt_s"] = float(wall_dt_s)
    measured = {
        "angle_rad": measured_angle_rad,
        "velocity_rad_s": measured_velocity_rad_s,
        "angular_acceleration_rad_s2": measured_angular_acceleration_rad_s2,
    }
    if any(value is not None for value in measured.values()):
        record["measured"] = {key: _vector_list(value) for key, value in measured.items() if value is not None}
    if prediction is not None:
        record["prediction"] = _prediction_record(prediction)
    if prediction_status is not None:
        record["prediction_status"] = str(prediction_status)
    return record


class RuleresultRecorderMixin:
    """给 ManagerBasedRLEnv 增加可选的逐步关节结果记录能力。

    这是 mixin（混入类），不单独创建环境。基础环境和模仿环境把它放在继承列表前面，
    因而可以拦截 reset、保存终止瞬间状态，并在正常 ``env.step`` 完成后生成输出。
    配置未启用 ``ruleresult_enabled`` 时所有记录路径都会快速返回，不影响训练动作维度。
    """

    def _ruleresult_init(self) -> None:
        """在基础环境已经存在后，按需初始化记录器缓存、计数器和文件句柄。

        延迟初始化让 mixin 不必修改环境构造函数，也避免只训练不导出时提前创建文件。
        ``previous_*`` 保存差分历史；``pending_*`` 暂存即将 reset 的终止帧。
        """

        if getattr(self, "_ruleresult_initialized", False):
            return
        self._ruleresult_initialized = True
        self._ruleresult_step = 0
        self._ruleresult_episode = 0
        self._ruleresult_previous_angle = None
        self._ruleresult_previous_velocity = None
        self._ruleresult_previous_sample_dt = None
        self._ruleresult_previous_timestamp = None
        self._ruleresult_previous_output_timestamp = None
        self._ruleresult_history_valid = None
        # ManagerBasedRLEnv 会在 ``step`` 内部重置已终止环境。这里保留一帧缓存，
        # 以便在重置状态覆盖 articulation 实测缓冲区之前，仍能输出真正的终止帧。
        self._ruleresult_pending_reset_mask = None
        self._ruleresult_pending_angle = None
        self._ruleresult_pending_velocity = None
        self._ruleresult_pending_acceleration = None
        self._ruleresult_pending_action = None
        self._ruleresult_pending_measured_angle = None
        self._ruleresult_latest = None
        self._ruleresult_latest_tensor = None
        self._ruleresult_latest_prediction = None
        self._ruleresult_prediction_callback = None
        self._ruleresult_stream = None
        self._ruleresult_output_path = None
        self._ruleresult_stream_format = None
        self._ruleresult_protocol_config = None
        self._ruleresult_loaded_protocol_config_path = None
        self._ruleresult_wall_clock_origin = time.perf_counter()
        self._ruleresult_previous_wall_timestamp = None
        self._ruleresult_previous_output_wall_timestamp = None

    @property
    def ruleresult(self) -> dict[str, Any] | None:
        """所选环境最近一行 JSON 安全记录；尚未记录时为 ``None``。"""

        self._ruleresult_init()
        return self._ruleresult_latest

    @property
    def ruleresult_tensor(self) -> torch.Tensor | None:
        """所有并行环境最近的目标结果张量 ``[N, 24]``；尚未记录时为 ``None``。"""

        self._ruleresult_init()
        return self._ruleresult_latest_tensor

    @property
    def ruleresult_prediction(self) -> dict[str, Any] | None:
        """所选环境最近的可选未来序列；未启用或不可用时为 ``None``。"""

        self._ruleresult_init()
        return self._ruleresult_latest_prediction

    def set_ruleresult_prediction_callback(self, callback: Any | None) -> None:
        """注册由外部提供未来位置目标的回调。

        调用形式为 ``callback(env, current_targets, horizon, dt_s)``。返回值可以是
        ``[N, H, 8]`` 张量（单环境也可用 ``[H, 8]``），也可以是已经含
        ``angles``/``timestamps_s`` 的字典。回调可接规划器或真正的多步预测模型；默认
        8维 PPO actor 没有未来动作头，所以没有回调时 ``supplied`` 模式会明确报告
        unavailable，而不会把当前动作重复 H 次伪装成预测。
        """

        self._ruleresult_init()
        if callback is not None and not callable(callback):
            raise TypeError("ruleresult prediction callback must be callable or None")
        self._ruleresult_prediction_callback = callback

    def _ruleresult_clear_history(
        self,
        increment_episode: bool = False,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        clear_time: bool = False,
    ) -> None:
        """清空全部或指定环境的差分历史，并按需更新 episode 编号。

        reset 后新姿态与上一 episode 末姿态不连续。如果保留旧角度，第一帧速度会变成
        ``(新初始角-旧终止角)/dt`` 的巨大假值，角加速度也会继续放大。因此至少把对应
        环境的 history_valid 设为假；完整 reset 还会清除时间戳历史。
        """

        self._ruleresult_init()
        if env_ids is None:
            self._ruleresult_previous_angle = None
            self._ruleresult_previous_velocity = None
            self._ruleresult_history_valid = None
            if self._ruleresult_pending_reset_mask is not None:
                self._ruleresult_pending_reset_mask.zero_()
        else:
            reference = self._ruleresult_history_valid
            if reference is None:
                reference = self._ruleresult_pending_reset_mask
            if reference is not None:
                if isinstance(env_ids, slice):
                    start, stop, step = env_ids.indices(reference.shape[0])
                    ids = torch.arange(start, stop, step, device=reference.device, dtype=torch.long)
                else:
                    ids = torch.as_tensor(env_ids, device=reference.device, dtype=torch.long).reshape(-1)
                if self._ruleresult_history_valid is not None:
                    self._ruleresult_history_valid[ids] = False
                if self._ruleresult_pending_reset_mask is not None:
                    self._ruleresult_pending_reset_mask[ids] = False
        if increment_episode:
            self._ruleresult_episode += 1
        if clear_time:
            self._ruleresult_previous_sample_dt = None
            self._ruleresult_previous_timestamp = None
            self._ruleresult_previous_output_timestamp = None
            self._ruleresult_previous_wall_timestamp = None
            self._ruleresult_previous_output_wall_timestamp = None
            self._ruleresult_latest_prediction = None

    def _ruleresult_control_dt_s(self) -> float:
        """返回标称策略控制周期 [s]，供固定时间模式和异常时钟兜底使用。"""

        configured = getattr(getattr(self, "cfg", None), "ruleresult_dt_s", CONTROL_DT_S)
        try:
            configured = float(configured)
        except (TypeError, ValueError):
            configured = CONTROL_DT_S
        step_dt = getattr(self, "step_dt", None)
        try:
            step_dt = float(step_dt)
        except (TypeError, ValueError):
            step_dt = None
        # 配置值若不是历史默认值，就视为用户显式覆盖。若它仍是历史默认值，而环境
        # 暴露了不同的步进周期，则优先采用管理器给出的权威周期。
        if step_dt is not None and math.isfinite(step_dt) and step_dt > 0.0:
            if math.isclose(configured, CONTROL_DT_S, rel_tol=0.0, abs_tol=1e-12) and not math.isclose(
                step_dt, CONTROL_DT_S, rel_tol=0.0, abs_tol=1e-12
            ):
                configured = step_dt
        if not math.isfinite(configured) or configured <= 0.0:
            raise ValueError(f"ruleresult control period must be positive and finite, got {configured}")
        return configured

    def _ruleresult_time_mode(self) -> str:
        """把别名归一化为 simulation、wall_clock 或 configured 时间模式。"""

        mode = str(getattr(getattr(self, "cfg", None), "ruleresult_time_mode", "simulation")).strip().lower()
        aliases = {"sim": "simulation", "wall": "wall_clock", "actual": "wall_clock"}
        mode = aliases.get(mode, mode)
        if mode not in RULERESULT_TIME_MODES:
            raise ValueError(f"ruleresult_time_mode must be one of {RULERESULT_TIME_MODES}, got {mode!r}")
        return mode

    def _ruleresult_sample_clock(self, control_dt_s: float) -> tuple[float, str, float, float | None]:
        """读取仿真/单调墙钟，并选出本次差分使用的时间戳。

        ``simulation`` 适合离线仿真，运行再慢也保持物理时间；``wall_clock`` 反映程序
        实际落地间隔，适合评估工控机调度抖动；``configured`` 始终按步数乘标称周期。
        返回值同时保留墙钟信息，JSONL 因而能比较物理时间和真实输出耗时。
        """

        wall_now = time.perf_counter() - self._ruleresult_wall_clock_origin
        previous_wall = getattr(self, "_ruleresult_previous_wall_timestamp", None)
        wall_dt = None if previous_wall is None else wall_now - previous_wall
        self._ruleresult_previous_wall_timestamp = wall_now
        mode = self._ruleresult_time_mode()
        if mode == "wall_clock":
            return wall_now, "wall_clock", wall_now, wall_dt
        if mode == "configured":
            return self._ruleresult_step * control_dt_s, "configured", wall_now, wall_dt
        sim_time = getattr(getattr(self, "sim", None), "current_time", None)
        try:
            sim_time = float(sim_time)
        except (TypeError, ValueError):
            sim_time = float("nan")
        if math.isfinite(sim_time):
            return sim_time, "simulation", wall_now, wall_dt
        return self._ruleresult_step * control_dt_s, "configured", wall_now, wall_dt

    def _ruleresult_position_target(self) -> torch.Tensor:
        """读取动作管理器处理后的绝对位置目标并重排为 ``[N, 8]`` [rad]。"""

        term = self.action_manager.get_term("joint_pos")
        target = _as_tensor(term.processed_actions)
        term_names = tuple(getattr(term, "_joint_names", ACTUATED_JOINT_NAMES))
        return _reorder_joint_tensor(target, term_names)

    def _ruleresult_measured_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """读取实际关节角、速度、角加速度，均为公共顺序的 ``[N, 8]`` 张量。"""

        robot = self.scene["robot"]
        names = tuple(getattr(robot, "joint_names", ACTUATED_JOINT_NAMES))
        data = robot.data
        return tuple(
            _reorder_joint_tensor(_as_tensor(getattr(data, field)), names)
            for field in ("joint_pos", "joint_vel", "joint_acc")
        )

    def _ruleresult_raw_action(self) -> torch.Tensor:
        """读取 actor 原始归一化动作，重排为公共顺序的 ``[N, 8]``。"""

        term = self.action_manager.get_term("joint_pos")
        action = getattr(term, "raw_actions", None)
        if action is None:
            action = self.action_manager.action
        action = _as_tensor(action)
        term_names = tuple(getattr(term, "_joint_names", ACTUATED_JOINT_NAMES))
        return _reorder_joint_tensor(action, term_names)

    @staticmethod
    def _ruleresult_env_ids(
        env_ids: Sequence[int] | torch.Tensor | slice, count: int, device: torch.device
    ) -> torch.Tensor:
        """把 reset 环境编号统一为目标设备上的一维整数张量。"""

        if isinstance(env_ids, slice):
            start, stop, step = env_ids.indices(count)
            return torch.arange(start, stop, step, device=device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=device, dtype=torch.long).reshape(-1)

    def _ruleresult_capture_before_reset(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """在 ManagerBasedRLEnv 覆盖状态前缓存终止帧。

        Isaac Lab 会在一次 ``step`` 内完成终止判断和自动 reset；当外层 ``step`` 返回
        时，实时关节缓冲区已经是新 episode 的初始姿态。这里提前保存目标、反馈和动作，
        让随后写出的 done 行仍表示真正终止瞬间，而不是误写 reset 后状态。
        """

        self._ruleresult_init()
        if not bool(getattr(getattr(self, "cfg", None), "ruleresult_enabled", False)):
            return
        try:
            target = self._ruleresult_position_target().detach()
            measured = tuple(value.detach() for value in self._ruleresult_measured_state())
            raw_action = self._ruleresult_raw_action().detach()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # 在构造阶段或 reset 收尾阶段，管理器可能尚未存在。正常 step 路径随后会
            # 自动回退到实时缓冲区，因此此处可以直接跳过快照。
            return
        if target.ndim != 2:
            return
        ids = self._ruleresult_env_ids(env_ids, target.shape[0], target.device)
        if ids.numel() == 0:
            return
        if not bool(((ids >= 0) & (ids < target.shape[0])).all()):
            raise ValueError("reset environment IDs are outside the ruleresult batch")
        needs_alloc = self._ruleresult_pending_reset_mask is None or tuple(
            self._ruleresult_pending_reset_mask.shape
        ) != (target.shape[0],)
        if needs_alloc:
            self._ruleresult_pending_reset_mask = torch.zeros(
                (target.shape[0],), dtype=torch.bool, device=target.device
            )
            self._ruleresult_pending_angle = torch.zeros_like(target)
            self._ruleresult_pending_velocity = torch.zeros_like(measured[1])
            self._ruleresult_pending_acceleration = torch.zeros_like(measured[2])
            self._ruleresult_pending_action = torch.zeros_like(raw_action)
        self._ruleresult_pending_reset_mask[ids] = True
        self._ruleresult_pending_angle[ids] = target[ids]
        self._ruleresult_pending_velocity[ids] = measured[1][ids]
        self._ruleresult_pending_acceleration[ids] = measured[2][ids]
        self._ruleresult_pending_action[ids] = raw_action[ids]
        # 终止角度保存的是目标轨迹值；实测位置单独缓存在另一字段中，避免混淆二者。
        if self._ruleresult_pending_measured_angle is None or tuple(
            self._ruleresult_pending_measured_angle.shape
        ) != tuple(measured[0].shape):
            self._ruleresult_pending_measured_angle = torch.zeros_like(measured[0])
        self._ruleresult_pending_measured_angle[ids] = measured[0][ids]

    def _reset_idx(self, env_ids: Sequence[int]):
        """先抓取指定环境的终止值，再调用基础环境完成 reset。"""

        self._ruleresult_capture_before_reset(env_ids)
        return super()._reset_idx(env_ids)

    def _ruleresult_output_file(self) -> Path:
        """解析输出路径；相对路径以运行日志目录或当前目录为基准。"""

        configured = getattr(self.cfg, "ruleresult_output_path", None)
        default_name = "ruleresult.bin" if self._ruleresult_output_format() == "protocol_55aa" else "ruleresult.jsonl"
        path = Path(configured).expanduser() if configured else Path(default_name)
        if not path.is_absolute():
            path = Path(getattr(self.cfg, "log_dir", None) or Path.cwd()) / path
        return path

    def _ruleresult_output_format(self) -> str:
        """返回规范化磁盘格式：可读 ``jsonl`` 或严格字节 ``protocol_55aa``。

        JSONL 每行自带字段名、单位、时间戳、反馈和可选未来序列，适合调试与离线分析；
        protocol_55aa 每周期只有一个 53B 电机命令帧，适合交给通讯层，但不携带时间戳、
        measured 反馈、schema 或预测序列。
        """

        output_format = str(getattr(self.cfg, "ruleresult_output_format", "jsonl")).strip().lower()
        aliases = {"json": "jsonl", "55aa": "protocol_55aa", "protocol-55aa": "protocol_55aa"}
        output_format = aliases.get(output_format, output_format)
        if output_format not in RULERESULT_OUTPUT_FORMATS:
            raise ValueError(f"ruleresult_output_format must be one of {RULERESULT_OUTPUT_FORMATS}")
        return output_format

    def _ruleresult_protocol_encoder(self) -> Protocol55AAConfig:
        """读取并缓存 55AA 电机标定配置；二进制导出必须显式提供该文件。"""

        configured = getattr(self.cfg, "ruleresult_protocol_config_path", None)
        if configured is None or not str(configured).strip():
            raise ValueError("ruleresult_protocol_config_path is required for protocol_55aa output")
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if self._ruleresult_protocol_config is None or self._ruleresult_loaded_protocol_config_path != path:
            self._ruleresult_protocol_config = Protocol55AAConfig.from_file(path)
            self._ruleresult_loaded_protocol_config_path = path
        return self._ruleresult_protocol_config

    def _ruleresult_prediction_for_batch(
        self,
        target: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        control_dt_s: float,
    ) -> tuple[dict[str, torch.Tensor] | None, str]:
        """按配置生成可选未来序列，并明确标注它是否真正可用。

        ``none`` 不输出未来；``kinematic`` 用当前 q/v/a 作恒加速度外推；``supplied``
        调用外部回调取得未来角度，并在缺少 v/a 时按未来时间戳求导。只有 supplied 的
        来源可能是实际规划器或多步模型，两种模式都不会自动多次调用当前 PPO actor。
        返回的批量序列形状为 ``[N, H, 8]``。
        """

        horizon = int(getattr(self.cfg, "ruleresult_prediction_horizon", 0))
        mode = str(getattr(self.cfg, "ruleresult_prediction_mode", "none")).strip().lower()
        if mode in {"constant_acceleration", "extrapolate"}:
            mode = "kinematic"
        if horizon <= 0 or mode == "none":
            return None, "disabled"
        if mode not in RULERESULT_PREDICTION_MODES:
            raise ValueError(f"ruleresult_prediction_mode must be one of {RULERESULT_PREDICTION_MODES}")
        prediction_dt = getattr(self.cfg, "ruleresult_prediction_dt_s", None)
        prediction_dt = control_dt_s if prediction_dt is None else float(prediction_dt)
        if not math.isfinite(prediction_dt) or prediction_dt <= 0.0:
            raise ValueError("ruleresult_prediction_dt_s must be positive and finite")
        if mode == "kinematic":
            prediction = extrapolate_ruleresult_sequence(
                target,
                velocity,
                acceleration,
                horizon_steps=horizon,
                dt_s=prediction_dt,
            )
            prediction["source"] = "constant_acceleration_extrapolation"
            return prediction, "available"

        callback = self._ruleresult_prediction_callback
        if callback is None:
            # 配置为 ``supplied`` 却没有提供序列时，应明确标记为不可用。若静默重复
            # 当前动作，会让调用方误以为这是真正的未来预测，并带来安全风险。
            return None, "unavailable"
        supplied = callback(self, target.detach(), horizon, prediction_dt)
        if isinstance(supplied, dict):
            prediction = dict(supplied)
            if "angles" not in prediction and "angle_targets_rad" in prediction:
                prediction["angles"] = prediction.pop("angle_targets_rad")
            prediction_angles = _as_tensor(prediction["angles"]).to(device=target.device, dtype=target.dtype)
            if prediction_angles.ndim == 2:
                prediction_angles = prediction_angles.unsqueeze(0).expand(target.shape[0], -1, -1)
            if (
                prediction_angles.ndim != 3
                or prediction_angles.shape[0] != target.shape[0]
                or prediction_angles.shape[1] != horizon
            ):
                raise ValueError("prediction callback angles must have shape [N, H, 8]")
            prediction["angles"] = prediction_angles
            for key in ("velocities", "accelerations"):
                if key in prediction:
                    value = _as_tensor(prediction[key]).to(device=target.device, dtype=target.dtype)
                    if value.ndim == 2:
                        value = value.unsqueeze(0).expand(target.shape[0], -1, -1)
                    prediction[key] = value
            if "timestamps_s" not in prediction:
                prediction["timestamps_s"] = (
                    torch.arange(1, horizon + 1, device=target.device, dtype=target.dtype) * prediction_dt
                )
            if "velocities" not in prediction or "accelerations" not in prediction:
                differentiated = predict_ruleresult_sequence(
                    prediction["angles"],
                    timestamps_s=prediction["timestamps_s"],
                    dt_s=prediction_dt,
                    initial_velocity_rad_s=velocity,
                )
                prediction.setdefault("velocities", differentiated["velocities"])
                prediction.setdefault("accelerations", differentiated["accelerations"])
                prediction.setdefault("dt_s", differentiated["dt_s"])
                prediction.setdefault("valid_mask", differentiated["valid_mask"])
        else:
            angle_sequence = _as_tensor(supplied).to(device=target.device, dtype=target.dtype)
            if angle_sequence.ndim == 2:
                angle_sequence = angle_sequence.unsqueeze(0)
                if target.shape[0] != 1:
                    angle_sequence = angle_sequence.expand(target.shape[0], -1, -1)
            if (
                angle_sequence.ndim != 3
                or angle_sequence.shape[0] != target.shape[0]
                or angle_sequence.shape[1] != horizon
            ):
                raise ValueError("prediction callback must return [N, H, 8] position targets")
            prediction = predict_ruleresult_sequence(
                angle_sequence,
                dt_s=prediction_dt,
                initial_velocity_rad_s=velocity,
            )
        prediction.setdefault("source", "supplied_future_targets")
        return prediction, "available"

    def _ruleresult_record_after_step(self, result: Any = None) -> None:
        """在一次环境步之后更新 ``[N, 24]`` 结果，并按设置写 JSONL 或 55AA。

        处理顺序是：读取目标和 done 掩码、还原已 reset 环境的终止快照、选择时间源、
        用非均匀 dt 计算导数、构造所选环境记录、可选生成未来序列、最后写盘并 flush。
        第一次样本、reset 后第一帧或时钟跳变后的第一帧没有可靠历史，导数置零并通过
        ``derivative_history_valid`` 告知读取者。
        """

        self._ruleresult_init()
        if not bool(getattr(self.cfg, "ruleresult_enabled", False)):
            return

        live_target = self._ruleresult_position_target().detach()
        if live_target.ndim != 2 or live_target.shape[-1] != len(ACTUATED_JOINT_NAMES):
            raise ValueError(f"expected a [num_envs, {len(ACTUATED_JOINT_NAMES)}] target tensor")
        env_count = live_target.shape[0]
        device = live_target.device
        done = torch.zeros((env_count,), dtype=torch.bool, device=device)
        if isinstance(result, tuple) and len(result) >= 4:
            terminated = torch.as_tensor(result[2], device=device, dtype=torch.bool).reshape(-1)
            truncated = torch.as_tensor(result[3], device=device, dtype=torch.bool).reshape(-1)
            if terminated.numel() != env_count or truncated.numel() != env_count:
                raise ValueError("ruleresult termination masks must match the environment batch")
            done = terminated | truncated

        # 执行到这里时，终止环境已经被基类重置。用 ``_reset_idx`` 提前保存的快照
        # 替换实时目标，使导出的这一行仍表示终止动作，而不是重置后的初始动作。
        target = live_target.clone()
        pending_mask = self._ruleresult_pending_reset_mask
        terminal_mask = done
        if pending_mask is not None and tuple(pending_mask.shape) == (env_count,):
            terminal_mask = terminal_mask & pending_mask.to(device=device)
            if bool(terminal_mask.any()):
                target[terminal_mask] = self._ruleresult_pending_angle[terminal_mask].to(device=device)

        control_dt_s = self._ruleresult_control_dt_s()
        self._ruleresult_step += 1
        current_timestamp, timestamp_source, wall_timestamp, wall_dt = self._ruleresult_sample_clock(control_dt_s)
        previous_timestamp = self._ruleresult_previous_timestamp
        timestamp_delta = None if previous_timestamp is None else current_timestamp - previous_timestamp
        min_dt_s = float(getattr(self.cfg, "ruleresult_min_dt_s", 1.0e-6))
        max_dt_s = float(getattr(self.cfg, "ruleresult_max_dt_s", 10.0))
        if not math.isfinite(min_dt_s) or min_dt_s <= 0.0:
            raise ValueError("ruleresult_min_dt_s must be positive and finite")
        if not math.isfinite(max_dt_s) or max_dt_s < min_dt_s:
            raise ValueError("ruleresult_max_dt_s must be finite and >= ruleresult_min_dt_s")
        clock_valid = timestamp_delta is None or (
            math.isfinite(timestamp_delta) and min_dt_s <= timestamp_delta <= max_dt_s
        )
        time_gap_detected = timestamp_delta is not None and not clock_valid
        # 有些仿真器包装层暴露的 current_time 不会递增。为兼容这种情况，时间差恰好
        # 为零时回退到配置周期；负时间差仍然视为无效。
        if timestamp_delta is not None and timestamp_delta == 0.0 and timestamp_source == "simulation":
            sample_dt_s = control_dt_s
            derivative_dt_source = "configured_fallback"
            clock_valid = True
        elif clock_valid and timestamp_delta is not None:
            sample_dt_s = float(timestamp_delta)
            derivative_dt_source = timestamp_source
        else:
            sample_dt_s = control_dt_s
            derivative_dt_source = f"{timestamp_source}_fallback"
        previous_sample_dt = self._ruleresult_previous_sample_dt
        self._ruleresult_previous_timestamp = current_timestamp
        self._ruleresult_previous_sample_dt = sample_dt_s
        previous_angle = self._ruleresult_previous_angle
        previous_velocity = self._ruleresult_previous_velocity
        history_valid = self._ruleresult_history_valid
        if (
            previous_angle is None
            or previous_velocity is None
            or history_valid is None
            or tuple(previous_angle.shape) != tuple(target.shape)
            or tuple(previous_velocity.shape) != tuple(target.shape)
            or tuple(history_valid.shape) != (env_count,)
        ):
            previous_angle = torch.zeros_like(target)
            previous_velocity = torch.zeros_like(target)
            history_valid = torch.zeros((env_count,), dtype=torch.bool, device=device)
        else:
            previous_angle = previous_angle.to(device=device, dtype=target.dtype)
            previous_velocity = previous_velocity.to(device=device, dtype=target.dtype)
            history_valid = history_valid.to(device=device)
        history_valid_before = history_valid.clone()
        # 每个并行环境独立计算导数。reset 后首帧或时钟中断后的首帧没有有效历史，
        # 因而从零速度、零加速度开始。时钟有抖动时，角加速度使用当前与上一采样间隔
        # 的平均值作为差分时间。
        derivative_history_valid = history_valid & bool(clock_valid)
        velocity = (target - previous_angle) / sample_dt_s
        velocity = torch.where(derivative_history_valid.unsqueeze(-1), velocity, torch.zeros_like(velocity))
        acceleration_dt_s = sample_dt_s
        if previous_sample_dt is not None and math.isfinite(float(previous_sample_dt)) and previous_sample_dt > 0.0:
            acceleration_dt_s = 0.5 * (sample_dt_s + float(previous_sample_dt))
        acceleration = (velocity - previous_velocity) / acceleration_dt_s
        acceleration = torch.where(derivative_history_valid.unsqueeze(-1), acceleration, torch.zeros_like(acceleration))
        self._ruleresult_previous_angle = target.detach().clone()
        self._ruleresult_previous_velocity = velocity.detach().clone()
        self._ruleresult_history_valid = torch.full((env_count,), bool(clock_valid), dtype=torch.bool, device=device)
        self._ruleresult_latest_tensor = flatten_ruleresult(target, velocity, acceleration).detach()

        interval = max(1, int(getattr(self.cfg, "ruleresult_interval", 1)))
        env_id = int(getattr(self.cfg, "ruleresult_env_id", 0))
        if not 0 <= env_id < target.shape[0]:
            raise ValueError(f"ruleresult_env_id={env_id} is outside [0, {target.shape[0]})")
        if self._ruleresult_step % interval == 0:
            measured_angle, measured_velocity, measured_acceleration = (
                value.detach().clone() for value in self._ruleresult_measured_state()
            )
            raw_action = self._ruleresult_raw_action().detach().clone()
            if bool(terminal_mask.any()) and pending_mask is not None:
                measured_angle[terminal_mask] = self._ruleresult_pending_measured_angle[terminal_mask].to(device=device)
                measured_velocity[terminal_mask] = self._ruleresult_pending_velocity[terminal_mask].to(device=device)
                measured_acceleration[terminal_mask] = self._ruleresult_pending_acceleration[terminal_mask].to(
                    device=device
                )
                raw_action[terminal_mask] = self._ruleresult_pending_action[terminal_mask].to(device=device)
            output_dt_s = None
            if self._ruleresult_previous_output_timestamp is not None:
                candidate = current_timestamp - self._ruleresult_previous_output_timestamp
                if math.isfinite(candidate) and candidate > 0.0:
                    output_dt_s = candidate
            output_wall_dt_s = None
            previous_output_wall = getattr(self, "_ruleresult_previous_output_wall_timestamp", None)
            if previous_output_wall is not None:
                candidate = wall_timestamp - previous_output_wall
                if math.isfinite(candidate) and candidate > 0.0:
                    output_wall_dt_s = candidate
            self._ruleresult_previous_output_timestamp = current_timestamp
            self._ruleresult_previous_output_wall_timestamp = wall_timestamp
            prediction_batch, prediction_status = self._ruleresult_prediction_for_batch(
                target, velocity, acceleration, control_dt_s
            )
            self._ruleresult_latest_prediction = None
            prediction_for_record = None
            if prediction_batch is not None:
                prediction_for_record = {
                    key: (value[env_id] if isinstance(value, torch.Tensor) and value.ndim >= 2 else value)
                    for key, value in prediction_batch.items()
                }
                self._ruleresult_latest_prediction = _prediction_record(prediction_for_record)
            self._ruleresult_latest = build_ruleresult_record(
                step=self._ruleresult_step,
                episode=self._ruleresult_episode,
                env_id=env_id,
                timestamp_s=current_timestamp,
                dt_s=sample_dt_s,
                control_dt_s=control_dt_s,
                output_dt_s=output_dt_s,
                dt_source=derivative_dt_source,
                derivative_valid=bool(clock_valid),
                time_gap_detected=time_gap_detected,
                wall_timestamp_s=wall_timestamp,
                wall_dt_s=output_wall_dt_s,
                angle_rad=target[env_id],
                velocity_rad_s=velocity[env_id],
                angular_acceleration_rad_s2=acceleration[env_id],
                measured_angle_rad=measured_angle[env_id],
                measured_velocity_rad_s=measured_velocity[env_id],
                measured_angular_acceleration_rad_s2=measured_acceleration[env_id],
                raw_action=raw_action[env_id],
                prediction=prediction_for_record,
                prediction_status=prediction_status,
                derivative_history_valid=bool(history_valid_before[env_id].item() and bool(clock_valid)),
            )
            output_path = self._ruleresult_output_file()
            output_format = self._ruleresult_output_format()
            if output_format == "jsonl":
                output_value = json.dumps(self._ruleresult_latest, ensure_ascii=False) + "\n"
            else:
                output_value = self._ruleresult_protocol_encoder().encode(
                    self._ruleresult_latest["joint_names"],
                    self._ruleresult_latest["angle_rad"],
                    self._ruleresult_latest["velocity_rad_s"],
                    self._ruleresult_latest["angular_acceleration_rad_s2"],
                )
            if (
                self._ruleresult_stream is None
                or self._ruleresult_output_path != output_path
                or self._ruleresult_stream_format != output_format
            ):
                if self._ruleresult_stream is not None:
                    self._ruleresult_stream.close()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_format == "jsonl":
                    self._ruleresult_stream = output_path.open("a", encoding="utf-8")
                else:
                    # 线上的二进制帧没有版本字段，因此绝不能把旧记录（包括 v1 小端帧）
                    # 与 v2 帧混写到同一个文件中。
                    if output_path.exists() and output_path.stat().st_size > 0:
                        raise ValueError(
                            f"refusing to append protocol frames to existing non-empty file {output_path}; "
                            "use a new output path"
                        )
                    self._ruleresult_stream = output_path.open("ab")
                self._ruleresult_output_path = output_path
                self._ruleresult_stream_format = output_format
                print(f"[INFO] Ruleresult output ({output_format}): {output_path}", flush=True)
            self._ruleresult_stream.write(output_value)
            self._ruleresult_stream.flush()

        if bool(done.any()):
            done_ids = done.nonzero(as_tuple=False).reshape(-1)
            self._ruleresult_clear_history(
                env_ids=done_ids,
                increment_episode=bool(done[env_id]),
            )
        # 待处理快照只对当前 step 有效。在这里清空它，避免后续未终止 step 误用旧状态。
        if self._ruleresult_pending_reset_mask is not None:
            self._ruleresult_pending_reset_mask.zero_()

    def reset(self, *args, **kwargs):
        """执行公开 reset，并为受影响环境开启全新的差分历史。"""

        requested_env_ids = kwargs.get("env_ids")
        # ManagerBasedRLEnv.reset(seed, env_ids, options) 把 env_ids 放在第二个位置参数。
        # 调用方只重置部分环境时，其余环境的差分历史必须保留。
        if requested_env_ids is None and len(args) >= 2:
            requested_env_ids = args[1]
        result = super().reset(*args, **kwargs)
        # 第一次公开 reset 从 episode 0 开始；后续手动 reset 才递增编号。同一 step 内
        # 自动发生的重置由 ``step`` 路径单独处理。
        has_previous_episode = self._ruleresult_step > 0 or self._ruleresult_latest is not None
        if requested_env_ids is None:
            self._ruleresult_clear_history(increment_episode=has_previous_episode, clear_time=True)
        else:
            env_id = int(getattr(self.cfg, "ruleresult_env_id", 0))
            increment_episode = False
            try:
                if isinstance(requested_env_ids, slice):
                    count = int(getattr(self, "num_envs", env_id + 1))
                    start, stop, step = requested_env_ids.indices(count)
                    increment_episode = env_id in range(start, stop, step)
                else:
                    ids = torch.as_tensor(requested_env_ids).reshape(-1)
                    increment_episode = bool((ids == env_id).any())
            except (TypeError, ValueError):
                increment_episode = has_previous_episode
            self._ruleresult_clear_history(increment_episode=increment_episode, env_ids=requested_env_ids)
        return result

    def reset_to(self, state, env_ids, seed=None, is_relative=False):
        """恢复外部指定的仿真状态，并使对应环境旧差分历史失效。"""

        result = super().reset_to(state, env_ids, seed=seed, is_relative=is_relative)
        has_previous_episode = self._ruleresult_step > 0 or self._ruleresult_latest is not None
        if env_ids is None:
            self._ruleresult_clear_history(increment_episode=has_previous_episode, clear_time=True)
        else:
            selected_env = int(getattr(self.cfg, "ruleresult_env_id", 0))
            try:
                ids = torch.as_tensor(env_ids).reshape(-1)
                increment_episode = bool((ids == selected_env).any())
            except (TypeError, ValueError):
                increment_episode = has_previous_episode
            self._ruleresult_clear_history(increment_episode=increment_episode, env_ids=env_ids)
        return result

    def close(self):
        """在释放仿真资源前关闭并清空结果文件句柄。"""

        self._ruleresult_init()
        if self._ruleresult_stream is not None:
            with contextlib.suppress(OSError):
                self._ruleresult_stream.close()
            self._ruleresult_stream = None
        return super().close()


__all__ = [
    "RULERESULT_DIM",
    "RULERESULT_OUTPUT_FORMATS",
    "RULERESULT_PREDICTION_MODES",
    "RULERESULT_SCHEMA",
    "RULERESULT_TIME_MODES",
    "RuleresultRecorderMixin",
    "build_ruleresult_record",
    "derive_target_kinematics",
    "extrapolate_ruleresult_sequence",
    "flatten_ruleresult",
    "predict_ruleresult_sequence",
]
