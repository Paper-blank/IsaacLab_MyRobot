# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""把仿真关节目标换算为实体电机侧目标。

本模块由 :mod:`deploy_prediction` 创建、由 :func:`deployment.replay.replay`
逐周期调用。输入 :class:`~deployment.trajectory.JointSample`，其中八个值使用项目
规定的仿真关节顺序；再结合 JSON 标定中的硬件编号、零位、方向和减速比，输出
:class:`MappedSample`。输出向量仍有八个元素，但顺序和数值已经是电机驱动器使用的
硬件侧约定。

阅读时可先看 :class:`JointCalibration` 中的换算参数，再看
:meth:`JointMapper.map_sample` 中的三条换算公式。这里不负责轨迹插值、安全限幅或
通信打包，这些工作分别属于 ``trajectory.py``、``safety.py`` 和实际总线适配器。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .trajectory import ACTUATED_JOINT_NAMES, JointSample


class MappingError(ValueError):
    """关节到电机的标定缺失、重复或数值无效时抛出的异常。"""


@dataclass(frozen=True)
class JointCalibration:
    """保存一个仿真关节到一个实体执行器的标定参数。

    角度换算公式为
    ``电机角度 = zero_offset_rad + direction * gear_ratio * 仿真关节角度``。
    速度和加速度没有零位偏移，只乘以 ``direction * gear_ratio``。这些参数必须
    在实体机构上测量和复核，程序不会从 URDF 自动推断它们。

    Attributes:
        hardware_id: 电机驱动器使用的唯一编号或名称。
        zero_offset_rad: 仿真关节角为零时对应的电机侧角度 [rad]。
        direction: 正方向关系，只允许 ``+1`` 或 ``-1``。
        gear_ratio: 从关节侧换算到电机侧的正减速比，无单位。
    """

    hardware_id: str
    zero_offset_rad: float = 0.0
    direction: float = 1.0
    gear_ratio: float = 1.0

    def __post_init__(self) -> None:
        """在不可变标定对象创建后立即拒绝空编号和无效数值。"""

        if not str(self.hardware_id):
            raise MappingError("hardware_id must not be empty")
        if self.direction not in (-1.0, 1.0):
            raise MappingError("direction must be +1 or -1")
        if not math.isfinite(float(self.zero_offset_rad)):
            raise MappingError("zero_offset_rad must be finite")
        if not math.isfinite(float(self.gear_ratio)) or self.gear_ratio <= 0.0:
            raise MappingError("gear_ratio must be positive and finite")


@dataclass(frozen=True)
class MappedSample:
    """一个已经换算到硬件顺序和电机侧物理量的完整采样点。

    Attributes:
        timestamp_s: 相对轨迹起点的时间 [s]。
        hardware_ids: 八个电机的唯一编号，shape 为 ``[8]``。
        angle_rad: 电机侧目标角度 [rad]，shape 为 ``[8]``。
        velocity_rad_s: 电机侧目标角速度 [rad/s]，shape 为 ``[8]``。
        acceleration_rad_s2: 电机侧目标角加速度 [rad/s^2]，shape 为 ``[8]``。
    """

    timestamp_s: float
    hardware_ids: tuple[str, ...]
    angle_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    acceleration_rad_s2: tuple[float, ...]


class JointMapper:
    """按固定关节顺序应用编号、零位、方向和减速比标定。

    映射器在初始化时就检查八个关节是否齐全以及硬件编号是否唯一。这样配置问题
    会在电机使能前暴露，而不是在回放中途才发现。
    """

    def __init__(
        self,
        calibrations: Mapping[str, JointCalibration] | Sequence[JointCalibration],
        *,
        source_names: Sequence[str] = ACTUATED_JOINT_NAMES,
    ) -> None:
        """创建并校验八关节映射。

        Args:
            calibrations: 可以是“仿真关节名 -> 标定”的字典，也可以是已经按照
                ``source_names`` 排好的标定序列。
            source_names: 输入轨迹的关节名。当前协议要求它严格等于项目的八关节
                规范顺序，shape 为 ``[8]``。

        Raises:
            MappingError: 关节顺序不正确、标定数量不为八、类型错误、缺少关节，
                或多个关节使用了同一个硬件编号。
        """

        self.source_names = tuple(str(name) for name in source_names)
        if self.source_names != tuple(ACTUATED_JOINT_NAMES):
            raise MappingError("source_names must use the canonical eight-joint order")
        if isinstance(calibrations, Mapping):
            try:
                ordered = [calibrations[name] for name in self.source_names]
            except KeyError as exc:
                raise MappingError(f"missing calibration for {exc.args[0]!r}") from exc
        else:
            ordered = list(calibrations)
        if len(ordered) != len(self.source_names) or not all(isinstance(item, JointCalibration) for item in ordered):
            raise MappingError("one JointCalibration is required per canonical joint")
        ids = tuple(str(item.hardware_id) for item in ordered)
        if len(set(ids)) != len(ids):
            raise MappingError("hardware_id values must be unique")
        self.calibrations = tuple(ordered)
        self.hardware_ids = ids

    @classmethod
    def identity(cls, source_names: Sequence[str] = ACTUATED_JOINT_NAMES) -> JointMapper:
        """创建不改变数值和名称的单位映射。

        该映射的零位为 ``0 rad``、方向为 ``+1``、减速比为 ``1``，主要用于空运行
        和测试。它不是实体机标定的替代品。

        Args:
            source_names: 八个规范关节名，shape 为 ``[8]``。

        Returns:
            每个仿真关节直接映射到同名硬件编号的映射器。
        """

        names = tuple(str(name) for name in source_names)
        return cls({name: JointCalibration(name) for name in names}, source_names=names)

    @classmethod
    def from_file(cls, path: str | Path, source_names: Sequence[str] = ACTUATED_JOINT_NAMES) -> JointMapper:
        """从 JSON 文件读取并校验八个关节的实体标定。

        文件顶层可以直接是关节字典，也可以把字典放在 ``joints`` 字段中。每个
        关节至少需要 ``hardware_id``，零位、方向和减速比缺省时分别使用
        ``0``、``+1`` 和 ``1``。

        Args:
            path: 标定 JSON 文件路径。
            source_names: 文件中必须包含的规范关节名，shape 为 ``[8]``。

        Returns:
            已按规范关节顺序排列并完成校验的映射器。

        Raises:
            MappingError: 文件不可读、JSON 语法错误、字段缺失或标定值不可转换。
        """

        try:
            payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MappingError(f"cannot read mapping file {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise MappingError("mapping file must contain a JSON object")
        entries = payload.get("joints", payload)
        if not isinstance(entries, Mapping):
            raise MappingError("mapping file needs a 'joints' object")
        mappings: dict[str, JointCalibration] = {}
        for name in tuple(source_names):
            item: Any = entries.get(name)
            if not isinstance(item, Mapping):
                raise MappingError(f"mapping is missing joint {name!r}")
            hardware_id = item.get("hardware_id", item.get("hardware_name"))
            if hardware_id is None:
                raise MappingError(f"mapping for {name!r} needs hardware_id")
            try:
                mappings[name] = JointCalibration(
                    str(hardware_id),
                    float(item.get("zero_offset_rad", 0.0)),
                    float(item.get("direction", item.get("sign", 1.0))),
                    float(item.get("gear_ratio", 1.0)),
                )
            except (TypeError, ValueError) as exc:
                raise MappingError(f"invalid mapping for {name!r}: {exc}") from exc
        return cls(mappings, source_names=source_names)

    def map_sample(self, sample: JointSample) -> MappedSample:
        """把一个八关节轨迹点换算为同一时刻的八电机目标。

        Args:
            sample: 仿真关节侧采样。角度、角速度、角加速度均为 shape ``[8]``。

        Returns:
            使用硬件编号和电机侧单位的采样；时间戳保持不变。

        Raises:
            MappingError: 采样中的关节数量与标定数量不一致。
        """

        if len(sample.angle_rad) != len(self.calibrations):
            raise MappingError("sample joint count does not match mapping")
        angles: list[float] = []
        velocities: list[float] = []
        accelerations: list[float] = []
        for value, velocity, acceleration, calibration in zip(
            sample.angle_rad,
            sample.velocity_rad_s,
            sample.acceleration_rad_s2,
            self.calibrations,
        ):
            scale = calibration.direction * calibration.gear_ratio
            angles.append(calibration.zero_offset_rad + scale * value)
            velocities.append(scale * velocity)
            accelerations.append(scale * acceleration)
        return MappedSample(
            sample.timestamp_s,
            self.hardware_ids,
            tuple(angles),
            tuple(velocities),
            tuple(accelerations),
        )

    # 保留早期部署脚手架使用的方法名，避免旧调用方立即失效。
    def apply(self, sample: JointSample) -> MappedSample:
        """调用 :meth:`map_sample`；这是兼容旧代码的别名。"""

        return self.map_sample(sample)

    @property
    def hardware_names(self) -> tuple[str, ...]:
        """返回八个硬件编号；这是 ``hardware_ids`` 的兼容别名。"""

        return self.hardware_ids


JointMapping = JointMapper


__all__ = [
    "JointCalibration",
    "JointMapper",
    "JointMapping",
    "MappedSample",
    "MappingError",
]
