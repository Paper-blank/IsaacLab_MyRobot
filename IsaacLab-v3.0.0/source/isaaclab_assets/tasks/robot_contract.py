# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""外骨骼任务在仿真、训练、结果导出和实体机之间共用的基础约定。

可以把本文件理解成整个外骨骼项目的“公共字典”：它不负责运行仿真，也不计算奖励，
只规定所有模块必须共同遵守的关节顺序、控制周期、观测维度和动作缩放值。把这些常量
集中在一个不依赖 Isaac Sim 的小文件里，可以避免训练环境认为第 1 个数是右转向关节，
而结果导出程序却把同一个数当成左踝关节之类的危险错位。

张量形状中常见的 ``N`` 表示并行环境数量。例如 ``[N, 8]`` 表示同时保存 N 台仿真
外骨骼的 8 个关节数据；最后一维始终按照 :data:`ACTUATED_JOINT_NAMES` 排列。
"""

ACTUATED_JOINT_NAMES = (
    "base_to_Rturn",
    "Rturn_to_Rthigh",
    "Rthigh_to_Rleg",
    "Rleg_to_Rfoot",
    "base_to_Lturn",
    "Lturn_to_Lthigh",
    "Lthigh_to_Lleg",
    "Lleg_to_Lfoot",
)
"""8 个主动关节的标准顺序：先右腿 4 个，再左腿 4 个。

训练策略的 8 维动作、关节角/速度张量以及 JSONL 结果均使用这个顺序。55AA 通讯协议
有自己按电机编号排列的顺序，协议编码器会按关节名称重新排序，不能直接假设二者下标
相同。
"""


IMU_SENSOR_NAMES = (
    "imu_rthigh",
    "imu_rleg",
    "imu_rfoot",
    "imu_lthigh",
    "imu_lleg",
    "imu_lfoot",
)
"""基础平地任务中 6 个可选连杆 IMU 的标准顺序。"""


CONTROL_DT_S = 0.02
"""策略控制周期 [s]：每 4 个 0.005 s 物理步更新一次，即标称 50 Hz。"""


POLICY_OBS_DIM = 49
"""基础任务给动作网络（actor）的观测特征数；单批形状为 ``[N, 49]``。"""


CRITIC_OBS_DIM = 62
"""价值网络（critic）的特权观测特征数；单批形状为 ``[N, 62]``。

“特权”表示这些额外信息可以在仿真训练时帮助估值，但实体机上的 actor 不需要获得它们。
"""


# 这些数值是策略动作的角度缩放 [rad]，不是经过认证的机械限位。网络动作通常先被
# RL-Games 限制在 [-1, 1]，随后环境按“默认关节角 + 动作 * 本缩放值”得到绝对位置
# 目标。它不是“当前关节角 + 增量”。膝、髋所需活动范围比转向和踝关节大，因此每个
# 关节分别配置。实体机仍必须使用根据真实机构测量得到的、更保守的软限位。
JOINT_ACTION_SCALE_RAD = {
    "base_to_Rturn": 0.45,
    "Rturn_to_Rthigh": 0.65,
    "Rthigh_to_Rleg": 0.95,
    "Rleg_to_Rfoot": 0.50,
    "base_to_Lturn": 0.45,
    "Lturn_to_Lthigh": 0.65,
    "Lthigh_to_Lleg": 0.95,
    "Lleg_to_Lfoot": 0.50,
}
"""8 个策略动作转换为关节位置目标时使用的逐关节缩放值 [rad]。"""


KNEE_JOINT_NAMES = ("Rthigh_to_Rleg", "Lthigh_to_Lleg")
"""右膝、左膝名称；顺序与公共关节顺序保持一致。"""


# URDF 中左右膝转轴是镜像的，所以相同正角度在两侧并不代表相同的人体屈膝方向。
# ``KNEE_ARTICULATION_SIGNS[name] * 仿真关节角`` 把两侧统一转换为“人体膝屈曲角”：
# 0 rad 约为伸直，正值表示屈膝。若以后更换 URDF 轴方向，只需改符号映射，不必重写
# 奖励函数和终止判断。
KNEE_ARTICULATION_SIGNS = {"Rthigh_to_Rleg": -1.0, "Lthigh_to_Lleg": 1.0}
KNEE_ANATOMICAL_LIMITS_RAD = (0.0, 2.2)
KNEE_EXTENSION_TOLERANCE_RAD = 0.05
KNEE_ARTICULATION_LIMITS_RAD = {
    name: (
        min(sign * KNEE_ANATOMICAL_LIMITS_RAD[0], sign * KNEE_ANATOMICAL_LIMITS_RAD[1]) - KNEE_EXTENSION_TOLERANCE_RAD,
        max(sign * KNEE_ANATOMICAL_LIMITS_RAD[0], sign * KNEE_ANATOMICAL_LIMITS_RAD[1]) + KNEE_EXTENSION_TOLERANCE_RAD,
    )
    for name, sign in KNEE_ARTICULATION_SIGNS.items()
}
"""人体屈曲范围 0..2.2 rad 换算回左右两侧原始仿真关节坐标后的限位。"""


__all__ = [
    "ACTUATED_JOINT_NAMES",
    "IMU_SENSOR_NAMES",
    "CONTROL_DT_S",
    "POLICY_OBS_DIM",
    "CRITIC_OBS_DIM",
    "JOINT_ACTION_SCALE_RAD",
    "KNEE_JOINT_NAMES",
    "KNEE_ARTICULATION_SIGNS",
    "KNEE_ANATOMICAL_LIMITS_RAD",
    "KNEE_EXTENSION_TOLERANCE_RAD",
    "KNEE_ARTICULATION_LIMITS_RAD",
]
