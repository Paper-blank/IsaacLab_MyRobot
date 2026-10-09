# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""外骨骼轨迹部署工具包的统一导入入口。

本包由 :mod:`scripts.deploy_prediction` 和将来的实体机控制程序调用。它输入离线
预测轨迹与关节标定，输出按固定周期排列、经过安全检查的整组电机命令。这里刻意
不依赖 CAN、EtherCAT、串口或 ROS 2；实体机接入前必须另外实现并审查
:class:`~deployment.motor_bus.MotorBus` 适配器。

初学者建议按数据流阅读：先看 :mod:`trajectory` 如何读取和插值，再看
:mod:`joint_mapping` 如何换算到电机侧，然后看 :mod:`safety`、
:mod:`motor_bus`，最后看 :mod:`replay` 如何把这些步骤组织成控制循环。本文件只把
常用名称重新导出，方便调用方从 ``deployment`` 一处导入，不包含控制逻辑。
"""

from .joint_mapping import JointCalibration, JointMapper, JointMapping, MappedSample, MappingError
from .motor_bus import DryRunBus, DryRunMotorBus, MotorBus, MotorCommand, MotorFrame, MotorState
from .replay import ReplayConfig, replay
from .safety import FeedbackState, JointLimit, SafetyConfig, SafetyLimiter, SafetyState, SafetyViolation
from .trajectory import (
    ACTUATED_JOINT_NAMES,
    DEFAULT_JOINT_NAMES,
    JointSample,
    Trajectory,
    TrajectoryError,
    load_trajectory,
)

__all__ = [
    "ACTUATED_JOINT_NAMES",
    "DEFAULT_JOINT_NAMES",
    "DryRunBus",
    "DryRunMotorBus",
    "FeedbackState",
    "JointCalibration",
    "JointLimit",
    "JointMapper",
    "JointMapping",
    "JointSample",
    "MappedSample",
    "MappingError",
    "MotorBus",
    "MotorCommand",
    "MotorFrame",
    "MotorState",
    "ReplayConfig",
    "SafetyConfig",
    "SafetyLimiter",
    "SafetyState",
    "SafetyViolation",
    "Trajectory",
    "TrajectoryError",
    "load_trajectory",
    "replay",
]
