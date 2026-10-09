# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""按单调时钟调度并发送八关节轨迹。

本模块是部署链路的总调度器，由 :mod:`scripts.deploy_prediction` 调用。输入包括
已经加载的 :class:`~deployment.trajectory.Trajectory`、关节映射、安全检查器和
电机总线；输出是成功发送的控制周期数量。每个周期依次执行“按时刻插值 -> 映射
到电机侧 -> 安全检查 -> 读取反馈 -> 发送整组命令”，绝不拆成单关节发送。

实时模式使用单调时钟（monotonic clock，不受系统日期校时影响）和绝对截止时间，
避免每周期的微小延迟不断累积。阅读时先看 :class:`ReplayConfig`，再从
:func:`replay` 的 ``try/finally`` 结构理解正常运行和异常停机的完整生命周期。
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .joint_mapping import JointMapper
from .motor_bus import MotorBus, MotorCommand
from .safety import SafetyLimiter, SafetyState, SafetyViolation
from .trajectory import JointSample, Trajectory


@dataclass(frozen=True)
class ReplayConfig:
    """回放循环的时序和控制模式配置。

    ``realtime=False`` 适合持续集成（CI）和快速空运行：循环按配置周期推进轨迹
    时间，但不等待现实时间。实体控制器应保持默认的实时模式，使用单调时钟和
    绝对截止时间。

    Attributes:
        control_period_s: 发送两组命令之间的目标周期 [s]；默认 ``0.02`` 即 50 Hz。
        interpolation: 轨迹点之间的插值方法，可选 ``linear``、``hermite3`` 或
            ``quintic``。
        mode: 写入 :class:`MotorCommand` 的控制模式，目前只允许 ``position`` 和
            ``position_velocity_ff``。
        max_lateness_s: 实际时间超过本周期绝对截止时间的最大容忍值 [s]。
        sleep_spin_threshold_s: 距截止时间小于该值时不再调用 ``sleep`` 的阈值 [s]，
            用于减少操作系统睡眠过头造成的抖动。
        realtime: 是否等待真实单调时钟；实体部署应为 ``True``。
    """

    control_period_s: float = 0.02
    interpolation: str = "linear"
    mode: str = "position"
    max_lateness_s: float = 0.05
    sleep_spin_threshold_s: float = 0.0005
    realtime: bool = True


def _sleep_until(deadline_ns: int, spin_threshold_s: float) -> None:
    """睡眠并短暂忙等，直到单调时钟到达绝对截止时间。

    Args:
        deadline_ns: :func:`time.monotonic_ns` 时钟域中的绝对截止时间 [ns]。
        spin_threshold_s: 距截止时间进入忙等阶段的阈值 [s]。

    Notes:
        这里使用“绝对时间”而不是每次固定睡眠一个周期。即使某一轮稍晚，下一轮
        仍瞄准原来的时间网格，不会把误差永久累加到后续周期。
    """

    while True:
        remaining_s = (deadline_ns - time.monotonic_ns()) / 1e9
        if remaining_s <= 0.0:
            return
        if remaining_s > spin_threshold_s:
            time.sleep(remaining_s - spin_threshold_s)


def _as_joint_sample(mapped: object) -> JointSample:
    """把映射结果转换为安全模块复用的关节采样结构。

    Args:
        mapped: 提供 ``timestamp_s``、``angle_rad``、``velocity_rad_s`` 和
            ``acceleration_rad_s2`` 属性的映射结果；三个向量 shape 均为 ``[8]``。

    Returns:
        数值仍处于电机侧、但类型统一为 :class:`JointSample` 的采样。
    """

    return JointSample(
        float(getattr(mapped, "timestamp_s")),
        tuple(getattr(mapped, "angle_rad")),
        tuple(getattr(mapped, "velocity_rad_s")),
        tuple(getattr(mapped, "acceleration_rad_s2")),
    )


def replay(
    trajectory: Trajectory,
    bus: MotorBus,
    mapping: JointMapper,
    safety: SafetyLimiter,
    config: ReplayConfig = ReplayConfig(),
    *,
    on_command: Callable[[MotorCommand], None] | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    """回放整条轨迹，并在每个控制周期发送一条完整八电机命令。

    回放循环不会发送部分关节更新。真实通信层必须把 :class:`MotorCommand` 转成
    厂商规定的 CAN、EtherCAT、串口或 ROS 2 消息，并通过 ``read_state`` 返回实际
    反馈。无论正常结束、用户中断还是发生异常，``finally`` 都会依次尝试停止、
    撤销使能并关闭总线。

    Args:
        trajectory: 已按时间排序的规范八关节轨迹，角度单位为 [rad]。
        bus: 实现 :class:`MotorBus` 的空运行或实体通信适配器。
        mapping: 从规范仿真关节顺序换算到八个硬件电机的映射器。
        safety: 对电机侧命令与反馈执行限位、超时和跟踪误差检查的安全器。
        config: 控制周期 [s]、插值方法、控制模式和时序容差。
        on_command: 可选回调；每成功发送一帧后调用，可用于日志或界面显示。
        stop_event: 可选线程事件；被置位时按用户中断路径安全退出。

    Returns:
        已成功发送的完整控制周期数量；轨迹终点至少发送一次。

    Raises:
        ValueError: 周期、迟到容差、控制模式或八关节数量不匹配。
        SafetyViolation: 截止时间、命令或反馈未通过安全检查。
        KeyboardInterrupt: 用户中断或 ``stop_event`` 请求停止。
    """

    if not math.isfinite(config.control_period_s) or config.control_period_s <= 0.0:
        raise ValueError("control_period_s must be positive and finite")
    if not math.isfinite(config.max_lateness_s) or config.max_lateness_s < 0.0:
        raise ValueError("max_lateness_s must be finite and non-negative")
    if config.mode not in {"position", "position_velocity_ff"}:
        raise ValueError("the scaffold supports position or position_velocity_ff; torque needs a dynamics controller")
    if trajectory.joint_count != safety.joint_count or trajectory.joint_count != len(mapping.source_names):
        raise ValueError("trajectory, mapping, and safety joint counts differ")

    period_ns = max(1, int(round(config.control_period_s * 1e9)))
    sequence = 0
    previous_hardware: JointSample | None = None
    try:
        bus.open()
        safety.arm()
        bus.enable()
        safety.start()
        # 电机使能可能耗时，不能把这段时间计入轨迹；首帧必须始终对应轨迹 t=0。
        start_ns = time.monotonic_ns()
        deadline_ns = start_ns
        while True:
            if stop_event is not None and stop_event.is_set():
                raise KeyboardInterrupt
            now_ns = time.monotonic_ns()
            if config.realtime:
                elapsed_s = max(0.0, (now_ns - start_ns) / 1e9)
                if sequence > 0 and now_ns - deadline_ns > int(config.max_lateness_s * 1e9):
                    raise SafetyViolation("control loop missed its absolute deadline")
            else:
                elapsed_s = sequence * config.control_period_s
            sample_time = min(elapsed_s, trajectory.duration_s)
            canonical = trajectory.sample(sample_time, method=config.interpolation)
            mapped = mapping.map_sample(canonical)
            hardware_sample = _as_joint_sample(mapped)
            dt_s = config.control_period_s if previous_hardware is not None else None
            safe_sample = safety.apply(hardware_sample, previous=previous_hardware, dt_s=dt_s)
            feedback = bus.read_state()
            # 发送新帧前读到的反馈通常对应上一帧目标。若拿它和即将发送的新目标
            # 比较，快速运动时会被误判为跟踪误差，因此这里选择上一周期目标。
            feedback_target = previous_hardware if previous_hardware is not None else safe_sample
            safety.check_feedback(feedback, feedback_target, now_ns=now_ns)
            frame = MotorCommand(
                sequence=sequence,
                timestamp_ns=now_ns,
                hardware_ids=mapped.hardware_ids,
                mode=config.mode,
                position_rad=safe_sample.angle_rad,
                velocity_rad_s=safe_sample.velocity_rad_s,
                acceleration_rad_s2=safe_sample.acceleration_rad_s2,
            )
            bus.send(frame)
            if on_command is not None:
                on_command(frame)
            previous_hardware = safe_sample
            sequence += 1
            # 轨迹终点只发送一次，随后由适配器进入其定义的安全状态；单点轨迹也
            # 会经过同一逻辑发送一次，而不是直接跳过。
            if elapsed_s >= trajectory.duration_s:
                break
            deadline_ns += period_ns
            if config.realtime:
                _sleep_until(deadline_ns, config.sleep_spin_threshold_s)
        safety.hold()
        return sequence
    except KeyboardInterrupt:
        safety.state = SafetyState.FAULT
        raise
    except Exception:
        safety.state = SafetyState.FAULT
        raise
    finally:
        try:
            bus.stop()
        finally:
            try:
                bus.disable()
            finally:
                bus.close()


__all__ = ["ReplayConfig", "replay"]
