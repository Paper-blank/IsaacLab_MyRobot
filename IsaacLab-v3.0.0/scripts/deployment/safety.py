# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""对电机侧命令、反馈和通信时效执行失效即停的安全检查。

本模块由 :func:`deployment.replay.replay` 在每个控制周期调用。输入是已经完成
关节标定的电机侧目标和总线反馈；输出要么是通过检查（或按配置限幅）的完整
八关节采样，要么抛出 :class:`SafetyViolation` 终止回放。“失效即停”
（fail-closed）表示信息缺失或超限时优先拒绝继续，而不是猜测一个值继续运行。

这里的软件检查只是实体安全链的一部分，不能替代机械限位、驱动器限流、独立急停、
硬件看门狗和上机前标定。默认配置没有具体关节限位，适合空运行验证；实体部署前
必须用经过测量和审批的 :class:`JointLimit` 与反馈策略替换默认值。

阅读顺序建议为：``JointLimit`` -> ``SafetyConfig`` -> ``SafetyLimiter.apply``
（检查命令）-> ``SafetyLimiter.check_feedback``（检查真实状态）。
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from .motor_bus import MotorState
from .trajectory import JointSample


class SafetyViolation(RuntimeError):
    """命令、反馈或时序不满足安全策略时抛出的异常。"""


class SafetyState(str, Enum):
    """描述安全器生命周期的有限状态。

    正常顺序是 ``INIT -> ARMED -> RUNNING -> HOLD``。主动停止可进入
    ``STOPPED``；命令、反馈、时序异常或用户中断会进入 ``FAULT``。状态只描述
    软件流程，真实驱动器是否断使能仍由 :class:`MotorBus` 适配器负责确认。
    """

    INIT = "init"
    ARMED = "armed"
    RUNNING = "running"
    HOLD = "hold"
    FAULT = "fault"
    STOPPED = "stopped"


@dataclass(frozen=True)
class JointLimit:
    """一个电机侧关节的可选软件限位。

    Attributes:
        position_min_rad: 允许的最小电机角度 [rad]；``None`` 表示本项未配置。
        position_max_rad: 允许的最大电机角度 [rad]；``None`` 表示本项未配置。
        velocity_max_rad_s: 允许的角速度绝对值上限 [rad/s]。
        acceleration_max_rad_s2: 允许的角加速度绝对值上限 [rad/s^2]。
        jerk_max_rad_s3: 相邻命令间角加速度变化率的绝对值上限 [rad/s^3]。

    Notes:
        这些值使用 :class:`JointMapper` 换算后的电机侧坐标，不一定等于 URDF 中的
        关节侧范围。``None`` 不是“自动安全”，而是“不执行该项检查”。
    """

    position_min_rad: float | None = None
    position_max_rad: float | None = None
    velocity_max_rad_s: float | None = None
    acceleration_max_rad_s2: float | None = None
    jerk_max_rad_s3: float | None = None

    def __post_init__(self) -> None:
        """拒绝非有限值、负幅值上限和颠倒的位置上下界。"""

        if self.position_min_rad is not None and not math.isfinite(self.position_min_rad):
            raise ValueError("position_min_rad must be finite")
        if self.position_max_rad is not None and not math.isfinite(self.position_max_rad):
            raise ValueError("position_max_rad must be finite")
        if self.position_min_rad is not None and self.position_max_rad is not None:
            if self.position_min_rad > self.position_max_rad:
                raise ValueError("position_min_rad must not exceed position_max_rad")
        for name in ("velocity_max_rad_s", "acceleration_max_rad_s2", "jerk_max_rad_s3"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class FeedbackState:
    """供测试和简单自定义适配器使用的最小反馈结构。

    Attributes:
        timestamp_ns: 采样时的单调时钟时间 [ns]。
        position_rad: 电机侧实际角度 [rad]，shape 为 ``[N]``。
        velocity_rad_s: 电机侧实际角速度 [rad/s]，shape 为 ``[N]``。
        fault: 驱动器是否报告故障。
        fault_message: 可选的故障说明。
        estop: 实体急停是否处于激活状态。

    ``N`` 必须等于安全器管理的关节数，本项目通常为八。完整实体适配器通常返回
    :class:`~deployment.motor_bus.MotorState`，其中还包含硬件编号顺序。
    """

    timestamp_ns: int
    position_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    fault: bool = False
    fault_message: str = ""
    estop: bool = False


@dataclass(frozen=True)
class SafetyConfig:
    """运行期软件安全策略；具体数值必须针对实体机构测量并审批。

    Attributes:
        limits: “硬件编号 -> 电机侧限位”的映射。``None`` 表示没有配置逐关节
            限位，只能用于空运行或额外安全层已经明确承担该职责的场景。
        feedback_timeout_s: 反馈相对当前单调时钟允许的最大数据年龄 [s]。
        tracking_error_max_rad: 每个硬件编号允许的“实际角度 - 目标角度”绝对值
            [rad]；``None`` 表示不检查跟踪误差。
        require_feedback: 为 ``True`` 时，任何周期缺少反馈都会立即终止。
        clip_velocity_and_acceleration: 是否把超限速度/加速度截到边界。默认
            ``False`` 会直接报错，便于在空运行阶段暴露坏轨迹。
    """

    limits: Mapping[str, JointLimit] | None = None
    feedback_timeout_s: float = 0.1
    tracking_error_max_rad: Mapping[str, float] | None = None
    require_feedback: bool = False
    clip_velocity_and_acceleration: bool = False

    def __post_init__(self) -> None:
        """保证反馈超时阈值为正且有限。"""

        if not math.isfinite(self.feedback_timeout_s) or self.feedback_timeout_s <= 0.0:
            raise ValueError("feedback_timeout_s must be positive and finite")


class SafetyLimiter:
    """在命令进入通信层之前校验每一组完整关节目标。

    位置超限和反馈故障始终终止循环。速度、加速度可以选择截断，但默认采用
    失效即停，以便在空运行或硬件在环（Hardware-in-the-Loop, HIL）阶段发现错误
    轨迹，而不是悄悄改变轨迹后继续执行。
    """

    def __init__(self, joint_names: Sequence[str], config: SafetyConfig | None = None) -> None:
        """创建一个按固定硬件顺序工作的安全检查器。

        Args:
            joint_names: 唯一硬件编号序列，shape 为 ``[N]``；本项目通常 ``N=8``。
            config: 限位、反馈超时和跟踪误差策略；不传时使用仅适合空运行检查的
                默认配置。

        Raises:
            ValueError: 硬件编号序列为空或存在重复项。
        """

        self.joint_names = tuple(str(name) for name in joint_names)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        self.config = config or SafetyConfig()
        self.state = SafetyState.INIT
        self._previous_acceleration: tuple[float, ...] | None = None

    @property
    def joint_count(self) -> int:
        """返回本安全器同时管理的硬件关节数量 ``N``。"""

        return len(self.joint_names)

    def arm(self) -> None:
        """把状态从初始化或已停止切换到已布防 ``ARMED``。"""

        if self.state not in {SafetyState.INIT, SafetyState.STOPPED}:
            raise SafetyViolation(f"cannot arm from {self.state.value}")
        self.state = SafetyState.ARMED

    def start(self) -> None:
        """把已布防状态切换到正在运行 ``RUNNING``。"""

        if self.state != SafetyState.ARMED:
            raise SafetyViolation(f"cannot start from {self.state.value}")
        self.state = SafetyState.RUNNING

    def hold(self) -> None:
        """在已布防或运行中把状态切换到保持 ``HOLD``。"""

        if self.state in {SafetyState.ARMED, SafetyState.RUNNING}:
            self.state = SafetyState.HOLD

    def stop(self) -> None:
        """把软件状态标记为已停止 ``STOPPED``。"""

        self.state = SafetyState.STOPPED

    def fault(self, reason: str) -> None:
        """把状态标记为故障并立即抛出安全异常。

        Args:
            reason: 供日志和操作者诊断的停止原因。

        Raises:
            SafetyViolation: 每次调用都会抛出。
        """

        self.state = SafetyState.FAULT
        raise SafetyViolation(reason)

    @staticmethod
    def _finite_vector(values: Sequence[float], expected: int, label: str) -> tuple[float, ...]:
        """把数值序列规范为元组，并同时检查长度和有限性。

        Args:
            values: 待检查的一维关节向量，shape 应为 ``[expected]``。
            expected: 期望元素数量。
            label: 出错时用于指出字段名称的文本。

        Returns:
            转换为 ``float`` 的不可变元组，shape 为 ``[expected]``。

        Raises:
            SafetyViolation: 长度不符，或包含 NaN/正负无穷大。
        """

        if len(values) != expected or any(not math.isfinite(float(value)) for value in values):
            raise SafetyViolation(f"{label} has the wrong length or a non-finite value")
        return tuple(float(value) for value in values)

    def _limit_for(self, index: int) -> JointLimit:
        """返回指定硬件槽位的限位，未配置时返回不设边界的默认值。

        Args:
            index: ``joint_names`` 中的零基下标。

        Returns:
            对应硬件编号的 :class:`JointLimit`。
        """

        limits = self.config.limits or {}
        return limits.get(self.joint_names[index], JointLimit())

    def apply(
        self,
        command: JointSample,
        *,
        previous: JointSample | None = None,
        dt_s: float | None = None,
    ) -> JointSample:
        """校验并按配置限制一个完整的电机侧命令。

        Args:
            command: 当前电机侧目标；角度、速度、加速度 shape 均为 ``[N]``。
            previous: 上一周期实际发送的安全目标。配置跃度上限时必须提供。
            dt_s: 当前与上一命令之间的时间 [s]；传入 ``previous`` 时必须为正。

        Returns:
            通过位置和跃度检查、并可能对速度/加速度限幅后的新
            :class:`JointSample`。原始 ``command`` 不会被原地修改。

        Raises:
            SafetyViolation: 向量数量或数值非法，位置超限，未启用截断时速度或
                加速度超限，或相邻命令计算出的跃度超限。

        Notes:
            位置从不自动截断，因为把不可达或危险位置悄悄改成边界值可能产生意外
            动作。跃度（jerk）按 ``(当前加速度 - 上一加速度) / dt_s`` 计算。
        """

        count = self.joint_count
        angle = self._finite_vector(command.angle_rad, count, "position")
        velocity = self._finite_vector(command.velocity_rad_s, count, "velocity")
        acceleration = self._finite_vector(command.acceleration_rad_s2, count, "acceleration")
        if previous is not None:
            if dt_s is None or not math.isfinite(dt_s) or dt_s <= 0.0:
                raise SafetyViolation("a positive dt_s is required for jerk limiting")
            if len(previous.acceleration_rad_s2) != count:
                raise SafetyViolation("previous command has the wrong joint count")
        for index, (position, speed, accel) in enumerate(zip(angle, velocity, acceleration)):
            limit = self._limit_for(index)
            if limit.position_min_rad is not None and position < limit.position_min_rad:
                raise SafetyViolation(f"{self.joint_names[index]} is below its position limit")
            if limit.position_max_rad is not None and position > limit.position_max_rad:
                raise SafetyViolation(f"{self.joint_names[index]} is above its position limit")
            if limit.velocity_max_rad_s is not None and abs(speed) > limit.velocity_max_rad_s:
                if self.config.clip_velocity_and_acceleration:
                    velocity = tuple(
                        max(-limit.velocity_max_rad_s, min(limit.velocity_max_rad_s, item))
                        if item_index == index
                        else item
                        for item_index, item in enumerate(velocity)
                    )
                else:
                    raise SafetyViolation(f"{self.joint_names[index]} exceeds its velocity limit")
            if limit.acceleration_max_rad_s2 is not None and abs(accel) > limit.acceleration_max_rad_s2:
                if self.config.clip_velocity_and_acceleration:
                    acceleration = tuple(
                        max(-limit.acceleration_max_rad_s2, min(limit.acceleration_max_rad_s2, item))
                        if item_index == index
                        else item
                        for item_index, item in enumerate(acceleration)
                    )
                else:
                    raise SafetyViolation(f"{self.joint_names[index]} exceeds its acceleration limit")
            if (
                previous is not None
                and limit.jerk_max_rad_s3 is not None
                and abs((acceleration[index] - previous.acceleration_rad_s2[index]) / dt_s) > limit.jerk_max_rad_s3
            ):
                raise SafetyViolation(f"{self.joint_names[index]} exceeds its jerk limit")
        result = JointSample(command.timestamp_s, angle, velocity, acceleration)
        self._previous_acceleration = acceleration
        return result

    def check_feedback(
        self,
        feedback: MotorState | FeedbackState | None,
        command: JointSample,
        *,
        now_ns: int | None = None,
    ) -> None:
        """校验驱动器反馈的故障、时效、顺序和位置跟踪误差。

        Args:
            feedback: 通信层最近反馈；当 ``require_feedback=False`` 时允许为
                ``None``。
            command: 与该反馈对应的电机侧目标，位置 shape 为 ``[N]``。回放循环
                通常传入上一周期目标，而不是尚未发送的新目标。
            now_ns: 当前单调时钟时间 [ns]；不传时在函数内读取。

        Raises:
            SafetyViolation: 必需反馈缺失、驱动器故障、急停激活、硬件顺序不同、
                向量非法、反馈过旧/来自过远未来，或位置跟踪误差超限。

        Notes:
            本函数只检查本通用数据结构包含的信息。电流、温度和厂商专用故障码等
            限制仍应由实体 :class:`MotorBus` 适配器实现。
        """

        if feedback is None:
            if self.config.require_feedback:
                raise SafetyViolation("motor feedback is required but unavailable")
            return
        if feedback.fault:
            raise SafetyViolation(feedback.fault_message or "motor drive reported a fault")
        if getattr(feedback, "estop", False):
            raise SafetyViolation("hardware emergency stop is active")
        feedback_ids = getattr(feedback, "hardware_ids", None)
        if feedback_ids is not None and tuple(str(item) for item in feedback_ids) != self.joint_names:
            raise SafetyViolation("feedback hardware order does not match the command order")
        positions = self._finite_vector(feedback.position_rad, self.joint_count, "feedback position")
        self._finite_vector(feedback.velocity_rad_s, self.joint_count, "feedback velocity")
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        age_s = (now_ns - int(feedback.timestamp_ns)) / 1e9
        if age_s < -self.config.feedback_timeout_s:
            raise SafetyViolation("motor feedback timestamp is in the future")
        if age_s > self.config.feedback_timeout_s:
            raise SafetyViolation(f"motor feedback is stale ({age_s:.3f}s)")
        if self.config.tracking_error_max_rad is not None:
            for index, name in enumerate(self.joint_names):
                bound = self.config.tracking_error_max_rad.get(name)
                if bound is not None and (not math.isfinite(bound) or bound < 0.0):
                    raise SafetyViolation(f"invalid tracking error limit for {name}")
                if bound is not None and abs(positions[index] - command.angle_rad[index]) > bound:
                    raise SafetyViolation(f"{name} tracking error exceeds its limit")

    # 保留初版部署脚手架中的名称，避免已有调用方立即失效。
    validate_command = apply


SafetyMonitor = SafetyLimiter
JointLimits = JointLimit


__all__ = [
    "FeedbackState",
    "JointLimit",
    "JointLimits",
    "SafetyConfig",
    "SafetyLimiter",
    "SafetyMonitor",
    "SafetyState",
    "SafetyViolation",
]
