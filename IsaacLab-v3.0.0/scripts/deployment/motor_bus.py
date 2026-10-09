# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""定义控制循环和实体通信适配器之间的统一接口。

:func:`deployment.replay.replay` 只和本模块定义的 :class:`MotorBus` 接口交互，
因此控制循环不需要知道底层使用 CAN、EtherCAT、串口还是 ROS 2。控制循环输入
:class:`MotorCommand`，总线适配器负责把它编码并原子发送给八个电机；适配器再用
:class:`MotorState` 返回同一硬件顺序的实际位置和速度。

本文件提供的 :class:`DryRunBus` 只在内存中保存命令，并把上一条命令模拟成反馈，
用于验证数据流和单元测试。它不会连接实体硬件。初学者可依次阅读三个数据结构、
``MotorBus`` 接口，再看 ``DryRunBus`` 如何实现这套接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MotorCommand:
    """一个控制周期内必须同时发送的整组电机命令。

    Attributes:
        sequence: 从零递增的控制周期序号，用于发现重复帧或丢帧。
        timestamp_ns: 生成命令时的单调时钟时间 [ns]，不是日期时间。
        hardware_ids: 电机编号，shape 为 ``[8]``；后面三个向量必须使用相同顺序。
        mode: 控制模式；当前回放层支持 ``position`` 或
            ``position_velocity_ff``（位置控制并提供速度前馈）。
        position_rad: 电机侧目标角度 [rad]，shape 为 ``[8]``。
        velocity_rad_s: 电机侧目标角速度 [rad/s]，shape 为 ``[8]``。
        acceleration_rad_s2: 电机侧目标角加速度 [rad/s^2]，shape 为 ``[8]``。

    Notes:
        本对象仍是物理量，不是 55AA 字节帧。真实适配器需要按驱动器协议执行
        量化、打包、校验和发送，并保证八个电机在同一控制周期更新。
    """

    sequence: int
    timestamp_ns: int
    hardware_ids: tuple[str, ...]
    mode: str
    position_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    acceleration_rad_s2: tuple[float, ...]


@dataclass(frozen=True)
class MotorState:
    """通信适配器返回的一次完整电机反馈。

    实际驱动器不一定提供电流或温度，因此可选遥测用 ``None`` 表示；位置和速度
    是参考安全检查必需的字段。所有关节向量都必须与 ``hardware_ids`` 顺序一致。

    Attributes:
        timestamp_ns: 采集反馈时的单调时钟时间 [ns]。
        hardware_ids: 电机编号，shape 为 ``[8]``。
        position_rad: 实际电机角度 [rad]，shape 为 ``[8]``。
        velocity_rad_s: 实际电机角速度 [rad/s]，shape 为 ``[8]``。
        current_a: 可选的实际电流 [A]，shape 为 ``[8]``。
        temperature_c: 可选的电机或驱动器温度 [degC]，shape 为 ``[8]``。
        fault: 驱动器是否报告故障。
        fault_message: 故障说明；无故障时为空字符串。
        estop: 实体急停（emergency stop）是否处于激活状态。
    """

    timestamp_ns: int
    hardware_ids: tuple[str, ...]
    position_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    current_a: tuple[float, ...] | None = None
    temperature_c: tuple[float, ...] | None = None
    fault: bool = False
    fault_message: str = ""
    estop: bool = False


class MotorBus(Protocol):
    """CAN、EtherCAT、串口或 ROS 2 实体适配器必须实现的接口。

    ``Protocol`` 是 Python 的结构化接口：类不必继承 ``MotorBus``，只要提供下列
    同名方法即可传给回放循环。实际实现仍须自行处理驱动器 SDK、超时、丢包和
    原子多关节更新。
    """

    def open(self) -> None:
        """打开通信资源，但此时不要使能电机。"""

        ...

    def close(self) -> None:
        """释放通信资源；该操作应允许在异常清理阶段重复调用。"""

        ...

    def enable(self) -> None:
        """使能所有受控电机，并确保它们准备接收整组命令。"""

        ...

    def disable(self) -> None:
        """撤销电机使能，具体制动策略由经审查的硬件适配器决定。"""

        ...

    def send(self, frame: MotorCommand) -> None:
        """把一个完整控制周期的命令原子发送到全部电机。"""

        ...

    def read_state(self) -> MotorState | None:
        """读取最近反馈；尚未收到有效反馈时返回 ``None``。"""

        ...

    def stop(self) -> None:
        """进入适配器定义的安全停止状态，不再接受正常轨迹命令。"""

        ...


class DryRunBus:
    """不连接实体接口、把上一条命令回显为反馈的内存总线。

    该类用于检查关节顺序、向量长度、生命周期和控制周期数量。它不能模拟通信
    延迟、跟踪误差、丢包或真实电机动力学，所以空运行成功不代表可以直接上机。
    """

    def __init__(self, hardware_ids: tuple[str, ...] | list[str]) -> None:
        """创建一个使用指定硬件顺序的空运行总线。

        Args:
            hardware_ids: 非空且互不重复的电机编号序列；本项目通常为八个元素。

        Raises:
            ValueError: 编号为空或存在重复项。
        """

        self.hardware_ids = tuple(str(item) for item in hardware_ids)
        if not self.hardware_ids or len(set(self.hardware_ids)) != len(self.hardware_ids):
            raise ValueError("dry-run hardware_ids must be non-empty and unique")
        self.enabled = False
        self.last_frame: MotorCommand | None = None
        self.frames_sent = 0

    def open(self) -> None:
        """模拟打开总线；空运行没有需要分配的外部资源。"""

        return None

    def close(self) -> None:
        """模拟关闭总线，并把使能状态清零。"""

        self.enabled = False

    def enable(self) -> None:
        """允许后续 :meth:`send` 接受模拟命令。"""

        self.enabled = True

    def disable(self) -> None:
        """禁止后续发送，模拟撤销电机使能。"""

        self.enabled = False

    def send(self, frame: MotorCommand) -> None:
        """校验并保存一条整组命令，不执行任何硬件通信。

        Args:
            frame: 硬件编号顺序和三个关节向量长度都必须与本总线一致的命令。

        Raises:
            RuntimeError: 总线尚未使能。
            ValueError: 硬件顺序不同或任一命令向量长度不正确。
        """

        if not self.enabled:
            raise RuntimeError("dry-run bus is disabled")
        if frame.hardware_ids != self.hardware_ids:
            raise ValueError("command hardware order does not match bus")
        count = len(self.hardware_ids)
        if any(
            len(values) != count
            for values in (frame.position_rad, frame.velocity_rad_s, frame.acceleration_rad_s2)
        ):
            raise ValueError("command vector length does not match hardware_ids")
        self.last_frame = frame
        self.frames_sent += 1

    def read_state(self) -> MotorState | None:
        """把最近命令的位置和速度回显为模拟反馈。

        Returns:
            还没有发送命令时返回 ``None``；否则返回与最近命令完全一致的位置、
            速度、时间戳和硬件顺序。
        """

        frame = self.last_frame
        if frame is None:
            return None
        return MotorState(
            timestamp_ns=frame.timestamp_ns,
            hardware_ids=self.hardware_ids,
            position_rad=frame.position_rad,
            velocity_rad_s=frame.velocity_rad_s,
        )

    def stop(self) -> None:
        """模拟安全停止，效果为立即撤销使能。"""

        self.enabled = False

    # 保留早期部署脚手架使用的方法名，便于旧调用方平滑迁移。
    def send_group(self, frame: MotorCommand) -> None:
        """调用 :meth:`send`；这是兼容旧代码的别名。"""

        self.send(frame)

    def read_feedback(self) -> MotorState | None:
        """调用 :meth:`read_state`；这是兼容旧代码的别名。"""

        return self.read_state()


DryRunMotorBus = DryRunBus
MotorFrame = MotorCommand
MotorFeedback = MotorState


__all__ = [
    "DryRunBus",
    "DryRunMotorBus",
    "MotorBus",
    "MotorCommand",
    "MotorFeedback",
    "MotorFrame",
    "MotorState",
]
