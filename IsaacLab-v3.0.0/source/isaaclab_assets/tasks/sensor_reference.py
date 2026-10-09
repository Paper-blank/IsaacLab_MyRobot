# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""为可选传感器模仿奖励提供驻留设备端的参考轨迹。

这个轻量通道用于已经整理成单条时间序列的参考数据，不在仿真进程里解析 ROS bag。
输入支持 NPZ、PyTorch、JSON 和 CSV；必需的八关节角以及可选速度、六路陀螺仪、
左右接触和 gait 标签会先在 CPU 上校验并重采样，然后一次性移动到仿真设备。
`joint_zero_rad` 和 `joint_sign` 只校正关节位置；文件自带的可选速度保持原值，不会自动乘
`joint_sign`。如果原始速度也在编码器坐标系中，调用方必须在入口前按同一方向规则转换。

``SensorReferenceBuffer`` 为每个并行环境保存一个独立整数游标 ``indices[N]``。episode
reset 可以只重置部分环境并选择随机起点；每个控制步统一前进，循环模式越过末尾回到
开头，非循环模式保持最后一帧。运行期间采样只做张量索引，不再访问文件系统。

该模块和完整多模态 ``imitation_dataset.py`` 用途不同：这里读取一个已整理参考文件；
后者负责从多层 trial 目录中同步 IMU、视频和事件记录。
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .robot_contract import ACTUATED_JOINT_NAMES, CONTROL_DT_S


class SensorReferenceFormatError(ValueError):
    """表示参考记录不满足字段、shape、时间或有限值契约。"""


@dataclass(frozen=True)
class SensorReference:
    """以 Torch 张量保存的一条时间对齐参考轨迹。

    所有字段第一维都是时间 ``T``，关节采用 canonical 右腿后左腿顺序。``t_s[T]``
    单位秒，``joint_pos_rad[T,8]`` 单位 rad；可选速度 ``[T,8]`` 和陀螺仪
    ``[T,6,3]`` 单位 rad/s，接触 ``[T,2]`` 归一化到 ``[0,1]``，gait 为 ``[T]``。
    """

    t_s: torch.Tensor
    joint_pos_rad: torch.Tensor
    joint_vel_rad_s: torch.Tensor | None = None
    gyro_rad_s: torch.Tensor | None = None
    contact_lr: torch.Tensor | None = None
    gait_id: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """返回参考轨迹的时间样本数 ``T``。"""

        return int(self.joint_pos_rad.shape[0])

    def sample(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """按每环境游标收集参考帧，并在轨迹末尾取模。

        Args:
            indices: 一维整数张量 ``[N]``，每项是一个环境的当前参考帧。

        Returns:
            字典中的必需关节角 shape 为 ``[N,8]``；其余只在源轨迹存在时返回，第一维
            同样为 ``N``。

        Raises:
            ValueError: ``indices`` 不是一维。
        """

        if indices.ndim != 1:
            raise ValueError(f"reference indices must be one-dimensional, got {tuple(indices.shape)}")
        safe_indices = indices.remainder(self.length)
        result = {
            "joint_pos_rad": self.joint_pos_rad[safe_indices],
        }
        for name in ("joint_vel_rad_s", "gyro_rad_s", "contact_lr", "gait_id"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value[safe_indices]
        return result


class SensorReferenceBuffer:
    """一条共享轨迹及每个并行环境独立的播放游标。"""

    def __init__(self, reference: SensorReference, device: torch.device, loop: bool = True):
        """把参考张量移到目标设备，并创建尚未分配环境数的游标。

        Args:
            reference: 至少两个时间样本的已校验轨迹。
            device: 训练环境使用的 CPU/CUDA 设备。
            loop: 是否循环播放；为 ``False`` 时游标最终停在末帧。

        Raises:
            SensorReferenceFormatError: 轨迹少于两个样本。
        """

        if reference.length < 2:
            raise SensorReferenceFormatError("reference trajectory must contain at least two samples")
        self.reference = _to_device(reference, device)
        self.device = device
        self.loop = bool(loop)
        self.indices = torch.zeros(0, dtype=torch.long, device=device)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        device: torch.device,
        target_dt_s: float = CONTROL_DT_S,
        loop: bool = True,
    ) -> SensorReferenceBuffer:
        """读取一条记录，按策略周期重采样并创建缓冲区。

        Args:
            path: NPZ、PT/PTH、JSON 或 CSV 文件。
            device: 缓冲区所在设备。
            target_dt_s: 策略控制周期 [s]。
            loop: 是否循环播放。

        Returns:
            已可供环境 reset/step 使用的缓冲区。
        """

        reference = load_sensor_reference(path, target_dt_s=target_dt_s)
        return cls(reference, device=device, loop=loop)

    @property
    def length(self) -> int:
        """返回共享参考轨迹的时间样本数 ``T``。"""

        return self.reference.length

    def reset(self, env_ids: torch.Tensor | slice, num_envs: int, random_start: bool = False) -> None:
        """只重置指定环境的游标，可为每个环境独立随机起点。

        Args:
            env_ids: 要重置的环境 ID 一维张量或 slice。
            num_envs: 当前并行环境总数；变化时会重新分配 ``indices[num_envs]``。
            random_start: ``True`` 时每个所选环境均匀随机选择 ``[0,T)`` 起点，否则归零。
        """

        if self.indices.numel() == 0 or self.indices.shape[0] != num_envs:
            self.indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, slice):
            ids = torch.arange(num_envs, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() > 0:
            if random_start:
                self.indices[ids] = torch.randint(self.length, (ids.numel(),), device=self.device)
            else:
                self.indices[ids] = 0

    def advance(self, steps: int = 1) -> None:
        """让全部已分配环境的游标前进若干策略样本。

        Args:
            steps: 非负整数步数。

        Raises:
            ValueError: ``steps`` 为负数。
        """

        if steps < 0:
            raise ValueError("reference playback steps must be non-negative")
        if self.indices.numel() == 0 or steps == 0:
            return
        if self.loop:
            self.indices.add_(steps).remainder_(self.length)
        else:
            self.indices.add_(steps).clamp_max_(self.length - 1)

    def sample(self) -> dict[str, torch.Tensor] | None:
        """返回全部环境当前参考帧；尚未 reset 分配游标时返回 ``None``。"""

        if self.indices.numel() == 0:
            return None
        return self.reference.sample(self.indices)


def load_sensor_reference(path: str | Path, target_dt_s: float = CONTROL_DT_S) -> SensorReference:
    """读取、标定、校验并重采样一条参考记录。

    支持 ``.npz``、``.pt/.pth``、``.json`` 和 CSV。必需字段是
    ``joint_pos_rad[T,8]``；可选字段为 ``t_s[T]``、速度 ``[T,8]``、陀螺仪
    ``[T,6,3]``、接触 ``[T,2]`` 和 gait ``[T]``。文件可附带八个
    ``joint_zero_rad`` 和 ``joint_sign`` 完成关节位置的编码器零位/方向标定。
    可选 ``joint_vel_rad_s`` 目前只做 shape、有限值与重采样检查，不会自动应用
    ``joint_sign``；速度的坐标约定必须由数据制作方事先统一。

    Args:
        path: 参考文件路径。
        target_dt_s: 输出时间间隔 [s]，连续量线性插值、gait 标签最近邻采样。

    Returns:
        时间从零开始、按 ``target_dt_s`` 重采样的 CPU :class:`SensorReference`。

    Raises:
        ValueError: ``target_dt_s`` 不为正数。
        FileNotFoundError: 文件不存在。
        SensorReferenceFormatError: 扩展名、字段、shape、时间顺序或数值不合法。
    """

    if target_dt_s <= 0.0:
        raise ValueError("target_dt_s must be positive")
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"sensor reference file does not exist: {path}")
    payload = _read_payload(path)
    reference = _reference_from_mapping(payload, target_dt_s=target_dt_s)
    return _resample_reference(reference, target_dt_s)


def _read_payload(path: Path) -> Mapping[str, Any]:
    """按扩展名读取文件为字段映射，暂不解释张量语义。

    PyTorch 文件优先使用 ``weights_only=True``，NPZ 禁止 pickle，避免为读取纯数值
    参考数据执行任意对象反序列化。

    Raises:
        SensorReferenceFormatError: 格式不支持或文件顶层不是映射/object。
    """

    suffix = path.suffix.lower()
    if suffix == ".npz":
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - Isaac Lab 通常已提供 NumPy
            raise SensorReferenceFormatError(".npz reference files require numpy") from exc
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if suffix in {".pt", ".pth"}:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # 兼容尚未提供 weights_only 参数的旧版 PyTorch
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise SensorReferenceFormatError(".pt reference files must contain a mapping")
        return payload
    if suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping):
            raise SensorReferenceFormatError(".json reference files must contain an object")
        return payload
    if suffix == ".csv":
        return _read_csv(path)
    raise SensorReferenceFormatError(f"unsupported reference extension '{path.suffix}'; use .npz, .pt, .json, or .csv")


def _read_csv(path: Path) -> Mapping[str, Any]:
    """读取数据准备指南约定的紧凑 CSV 列格式。

    关节列支持 ``field[0]``、``field_0`` 和 ``field0`` 三种编号形式；六路三轴陀螺仪
    在 CSV 中铺平成 18 列，返回映射前恢复为 ``[T,6,3]``。

    Raises:
        SensorReferenceFormatError: 文件无数据行或缺少八个必需关节角列。
        ValueError: 某个存在的数值单元无法转换为浮点数。
    """

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SensorReferenceFormatError("CSV reference file has no data rows")
    columns = list(rows[0])

    def column(*names: str) -> list[float] | None:
        """读取首个存在的 CSV 列并转为浮点列表；所有别名缺失时返回 ``None``。"""

        for name in names:
            if name in columns:
                return [float(row[name]) for row in rows]
        return None

    result: dict[str, Any] = {"t_s": column("t_s", "time_s", "time")}
    q_columns = _numbered_columns(columns, ("joint_pos_rad", "q_rad", "q"), len(ACTUATED_JOINT_NAMES))
    if q_columns is None:
        raise SensorReferenceFormatError("CSV must provide joint_pos_rad[0..7] (or q_rad[0..7]) columns")
    result["joint_pos_rad"] = [[float(row[name]) for name in q_columns] for row in rows]
    for field, prefixes, count in (
        ("joint_vel_rad_s", ("joint_vel_rad_s", "qdot_rad_s", "qdot"), 8),
        ("contact_lr", ("contact_lr", "contact"), 2),
    ):
        names = _numbered_columns(columns, prefixes, count)
        if names is not None:
            result[field] = [[float(row[name]) for name in names] for row in rows]
    gyro_names = _numbered_columns(columns, ("gyro_rad_s", "gyro"), 18)
    if gyro_names is not None:
        result["gyro_rad_s"] = [
            [[float(row[name]) for name in gyro_names[i : i + 3]] for i in range(0, 18, 3)] for row in rows
        ]
    return {key: value for key, value in result.items() if value is not None}


def _numbered_columns(columns: list[str], prefixes: tuple[str, ...], count: int) -> list[str] | None:
    """寻找从 0 连续编号的整组列名；任一编号缺失时尝试下一个前缀。"""

    for prefix in prefixes:
        candidates = []
        for index in range(count):
            names = (f"{prefix}[{index}]", f"{prefix}_{index}", f"{prefix}{index}")
            match = next((name for name in names if name in columns), None)
            if match is None:
                break
            candidates.append(match)
        if len(candidates) == count:
            return candidates
    return None


def _reference_from_mapping(payload: Mapping[str, Any], target_dt_s: float) -> SensorReference:
    """把别名兼容的字段映射规范为严格的 :class:`SensorReference`。

    时间缺失时按 ``target_dt_s`` 生成；关节零位先减、方向符号后乘。接触值绝对值
    大于 1 时视为力值并按 5 N 归一化，再统一裁剪到 ``[0,1]``。

    Returns:
        尚未重采样、但字段 shape、有限值和时间单调性均已验证的参考对象。

    Raises:
        SensorReferenceFormatError: 必需字段缺失，任一字段 shape/标定/时间契约不合法。
    """

    def get(*names: str) -> Any:
        """从输入映射读取首个字段别名；全部缺失时返回 ``None``。"""

        for name in names:
            if name in payload:
                return payload[name]
        return None

    joint_pos = _tensor(get("joint_pos_rad", "joint_pos", "q_rad", "q"), "joint_pos_rad")
    if joint_pos.ndim != 2:
        raise SensorReferenceFormatError(f"joint_pos_rad must have shape [T, 8], got {tuple(joint_pos.shape)}")
    if joint_pos.shape[-1] != len(ACTUATED_JOINT_NAMES) and joint_pos.shape[0] == len(ACTUATED_JOINT_NAMES):
        joint_pos = joint_pos.transpose(0, 1)
    if joint_pos.shape[-1] != len(ACTUATED_JOINT_NAMES):
        raise SensorReferenceFormatError(f"joint_pos_rad must have shape [T, 8], got {tuple(joint_pos.shape)}")

    zero = get("joint_zero_rad", "zero_rad")
    sign = get("joint_sign", "sign")
    if zero is not None:
        zero_tensor = _tensor(zero, "joint_zero_rad").reshape(-1)
        if zero_tensor.numel() != joint_pos.shape[-1]:
            raise SensorReferenceFormatError("joint_zero_rad must contain eight values")
        joint_pos = joint_pos - zero_tensor
    if sign is not None:
        sign_tensor = _tensor(sign, "joint_sign").reshape(-1)
        if sign_tensor.numel() != joint_pos.shape[-1] or not torch.all(torch.abs(sign_tensor) == 1):
            raise SensorReferenceFormatError("joint_sign must contain eight values, each +1 or -1")
        joint_pos = joint_pos * sign_tensor

    length = joint_pos.shape[0]
    t_value = get("t_s", "time_s", "time", "timestamps_s")
    if t_value is None:
        t_s = torch.arange(length, dtype=torch.float32) * target_dt_s
    else:
        t_s = _tensor(t_value, "t_s").reshape(-1)
        if t_s.numel() != length:
            raise SensorReferenceFormatError("t_s length must match joint_pos_rad")
        t_s = t_s - t_s[0]
    if length < 2 or not torch.isfinite(joint_pos).all() or not torch.isfinite(t_s).all():
        raise SensorReferenceFormatError("reference time and joint positions must be finite and contain >=2 samples")
    if torch.any(torch.diff(t_s) <= 0):
        raise SensorReferenceFormatError("t_s must be strictly increasing")

    joint_vel = _optional_tensor(get("joint_vel_rad_s", "joint_vel", "qdot_rad_s", "qdot"), length, (8,))
    gyro = _optional_tensor(get("gyro_rad_s", "gyro", "imu_gyro_rad_s"), length, (6, 3))
    contact = _optional_tensor(get("contact_lr", "contact", "foot_contact"), length, (2,))
    gait = _optional_tensor(get("gait_id", "gait"), length, ())
    if contact is not None:
        # 二值接触和类似力的记录都可接受；绝对值大于 1 时按 N 理解，
        # 再用 5 N 归一化到 [0, 1]。
        contact = torch.where(contact.abs() > 1.0, contact.abs() / 5.0, contact).clamp(0.0, 1.0)
    if gait is not None:
        gait = gait.round().to(dtype=torch.long).reshape(length)
    return SensorReference(t_s, joint_pos, joint_vel, gyro, contact, gait)


def _tensor(value: Any, name: str) -> torch.Tensor:
    """把必需数值字段转为 CPU ``float32`` 张量，缺失或非数值时抛出格式错误。"""

    if value is None:
        raise SensorReferenceFormatError(f"reference is missing required field '{name}'")
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise SensorReferenceFormatError(f"reference field '{name}' is not numeric") from exc
    return tensor


def _optional_tensor(value: Any, length: int, trailing_shape: tuple[int, ...]) -> torch.Tensor | None:
    """校验可选时间序列字段，并兼容铺平的 ``[T,18]`` 陀螺仪。"""

    if value is None:
        return None
    tensor = _tensor(value, "optional")
    expected = (length, *trailing_shape)
    if tensor.shape != expected:
        # 同时接受已展平的 [T, 18] 陀螺仪输入。
        if trailing_shape == (6, 3) and tensor.shape == (length, 18):
            tensor = tensor.reshape(length, 6, 3)
        else:
            raise SensorReferenceFormatError(f"optional field must have shape {expected}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise SensorReferenceFormatError("optional reference fields must be finite")
    return tensor


def _resample_reference(reference: SensorReference, target_dt_s: float) -> SensorReference:
    """连续通道线性重采样，离散 gait 标签最近邻重采样。

    Args:
        reference: 时间严格递增的源轨迹。
        target_dt_s: 目标样本间隔 [s]。

    Returns:
        从 0 秒开始的新轨迹；位置、速度、陀螺仪和接触连续插值，接触再次裁剪到
        ``[0,1]``，gait 保持整数标签。
    """

    old_t = reference.t_s
    duration = float(old_t[-1])
    count = max(2, int(torch.floor(torch.tensor(duration / target_dt_s)).item()) + 1)
    new_t = torch.arange(count, dtype=torch.float32) * target_dt_s
    new_t[-1] = min(float(new_t[-1]), duration)

    def interpolate(values: torch.Tensor) -> torch.Tensor:
        """把连续张量沿时间维线性插值到 ``new_t``，其余维度保持不变。"""

        right = torch.searchsorted(old_t, new_t).clamp(1, old_t.numel() - 1)
        left = right - 1
        alpha = ((new_t - old_t[left]) / (old_t[right] - old_t[left])).reshape(-1, *([1] * (values.ndim - 1)))
        return values[left] * (1.0 - alpha) + values[right] * alpha

    def nearest(values: torch.Tensor) -> torch.Tensor:
        """按最近时间点采样离散标签，距离相等时选择左侧旧样本。"""

        right = torch.searchsorted(old_t, new_t).clamp(1, old_t.numel() - 1)
        left = right - 1
        choose_right = (new_t - old_t[left]) > (old_t[right] - new_t)
        return values[torch.where(choose_right, right, left)]

    return SensorReference(
        new_t,
        interpolate(reference.joint_pos_rad),
        None if reference.joint_vel_rad_s is None else interpolate(reference.joint_vel_rad_s),
        None if reference.gyro_rad_s is None else interpolate(reference.gyro_rad_s),
        None if reference.contact_lr is None else interpolate(reference.contact_lr).clamp(0.0, 1.0),
        None if reference.gait_id is None else nearest(reference.gait_id),
    )


def _to_device(reference: SensorReference, device: torch.device) -> SensorReference:
    """把存在的所有参考张量移动到同一设备，保持缺失字段为 ``None``。"""

    return SensorReference(
        *(
            value.to(device=device) if value is not None else None
            for value in (
                reference.t_s,
                reference.joint_pos_rad,
                reference.joint_vel_rad_s,
                reference.gyro_rad_s,
                reference.contact_lr,
                reference.gait_id,
            )
        )
    )


__all__ = [
    "SensorReference",
    "SensorReferenceBuffer",
    "SensorReferenceFormatError",
    "load_sensor_reference",
]
