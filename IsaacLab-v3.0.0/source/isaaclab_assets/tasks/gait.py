# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""外骨骼运动任务共用的步态编号和课程学习阶段。

这里的“步态”不是又一个神经网络，而是环境维护的状态标签。标签会作为 one-hot 特征
放进策略观测，也会决定某一组奖励此刻是否生效。例如站立奖励只在 ``STAND`` 环境中
计分，行走摆脚奖励只在 ``WALK`` 环境中计分。

课程学习（Curriculum Learning）是先教简单任务、再逐渐增加难度：早期只练行走，
中期加入起停转换，后期才开放跑步等多步态命令。本文件只定义编号和判断小工具，具体
从第几步切换由环境配置决定。
"""

from __future__ import annotations

from enum import IntEnum

import torch


class GaitMode(IntEnum):
    """策略可见的 5 种步态标签，观测中编码为长度 5 的 one-hot 向量。"""

    STAND = 0
    WALK = 1
    WALK_TO_STAND = 2
    RUN = 3
    RUN_TO_WALK = 4


class CurriculumPhase(IntEnum):
    """训练难度逐步提升时使用的 3 个课程阶段。"""

    WALK_ONLY = 1
    TRANSITIONS = 2
    MULTI_GAIT = 3


GAIT_MODE_COUNT = len(GaitMode)
"""步态 one-hot 特征数，目前为 5。"""

FROUDE_THRESHOLD = 0.5
"""无量纲 Froude 数阈值；超过它时可把高速运动归类为跑步。"""

LEG_LENGTH_M = 0.8
"""计算 Froude 数时使用的等效腿长 [m]。"""

GRAVITY_M_S2 = 9.81
"""计算 Froude 数时使用的标准重力加速度 [m/s^2]。"""

WALK_TO_STAND_SPEED_M_S = 0.1
"""命令速度低于该值时进入“行走转站立”判断 [m/s]。"""

WALK_SPEED_M_S = 0.2
"""转换阶段奖励使用的参考行走速度 [m/s]。"""

MAX_CURRICULUM_SPEED_M_S = 4.0
"""课程最后阶段允许采样的最大前进命令速度 [m/s]。"""


def froude_number(speed_m_s: float, leg_length_m: float = LEG_LENGTH_M) -> float:
    """根据前进速度计算 Froude 数 ``速度^2 / (重力加速度 * 腿长)``。"""

    if leg_length_m <= 0.0:
        raise ValueError("leg_length_m must be positive")
    return speed_m_s * speed_m_s / (GRAVITY_M_S2 * leg_length_m)


def curriculum_phase_for_step(step: int, phase2_start: int, phase3_start: int) -> CurriculumPhase:
    """根据全局控制步数返回当前课程阶段。

    Args:
        step: 从训练开始累计的环境控制步数，不是 PPO 的 epoch 编号。
        phase2_start: 从该控制步开始开放起停转换。
        phase3_start: 从该控制步开始开放跑步和完整多步态。
    """

    if phase2_start < 0 or phase3_start < phase2_start:
        raise ValueError("curriculum phase boundaries must be non-negative and ordered")
    if step >= phase3_start:
        return CurriculumPhase.MULTI_GAIT
    if step >= phase2_start:
        return CurriculumPhase.TRANSITIONS
    return CurriculumPhase.WALK_ONLY


def gait_mask(env, gait: GaitMode | int) -> torch.Tensor:
    """为指定步态生成形状 ``[N]`` 的 0/1 掩码。

    当前属于该步态的并行环境取 1，否则取 0。奖励乘上此掩码后，只会作用于正确的
    步态，避免站立环境误拿到行走奖励。
    """

    command = env.command_manager.get_term("base_velocity")
    if not hasattr(command, "gait_mode"):
        raise TypeError("base_velocity must expose a gait_mode tensor")
    return (command.gait_mode == int(gait)).to(dtype=torch.float32)


__all__ = [
    "CurriculumPhase",
    "FROUDE_THRESHOLD",
    "GAIT_MODE_COUNT",
    "GRAVITY_M_S2",
    "GaitMode",
    "LEG_LENGTH_M",
    "MAX_CURRICULUM_SPEED_M_S",
    "WALK_SPEED_M_S",
    "WALK_TO_STAND_SPEED_M_S",
    "curriculum_phase_for_step",
    "froude_number",
    "gait_mask",
]
