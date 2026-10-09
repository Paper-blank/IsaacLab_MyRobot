"""外骨骼平地运动任务的完整 Isaac Lab 配置。

本文件主要回答“环境里有什么、策略能看到什么、动作怎样执行、怎样打分、何时重置”这
几类问题。Isaac Lab 使用多个 ``*Cfg`` 配置类分层组织它们：

* :class:`MyExoskeletonSceneCfg` 创建机器人、地面、灯光和传感器；
* :class:`MyExoskeletonCommandsCfg` 产生期望速度和步态命令；
* :class:`MyExoskeletonObservationsCfg` 决定 actor/critic 看见哪些数；
* :class:`MyExoskeletonActionsCfg` 把网络的 8 维动作转换为关节位置目标；
* :class:`MyExoskeletonRewardsCfg` 和 :class:`MyExoskeletonTerminationsCfg`
  分别定义奖励与 episode 结束条件；
* :class:`MyExoskeletonEventsCfg` 和 :class:`MyExoskeletonCurriculumCfg`
  定义物理随机化、扰动以及训练难度的推进；
* :class:`MyExoskeletonFlatEnvCfg` 最后把以上配置组装成一个可创建的环境。

URDF 含 8 个主动旋转关节、腿部 6 个 IMU 和左右脚各 1 个接触传感器。配置阶段只描述
它们；真正的仿真对象会在 ``gym.make(...)`` 创建环境时生成。
"""

from __future__ import annotations

import math
from pathlib import Path

from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity import mdp

from isaaclab_assets.tasks.robot_contract import (
    ACTUATED_JOINT_NAMES,
    CONTROL_DT_S,
    CRITIC_OBS_DIM,
    JOINT_ACTION_SCALE_RAD,
    KNEE_ARTICULATION_LIMITS_RAD,
    KNEE_JOINT_NAMES,
    POLICY_OBS_DIM,
)

# 这些传感器实现依赖已经启动的 Omniverse Kit。这里使用“模块路径:类名”的字符串，
# 让查看配置、执行 --help 或运行不启动 Kit 的单元测试时，不会过早导入仿真库。
_INSPECTABLE_IMU_CLASS = "isaaclab_assets.sensors.inspectable_imu:InspectableImu"
_NESTED_CONTACT_SENSOR_CLASS = "isaaclab_assets.sensors.contact_sensor:NestedBodyContactSensor"
_TASK_MODULE = "isaaclab_assets.tasks.my_exoskeleton_flat"


def _task_term(name: str) -> str:
    """把函数名变成 Isaac Lab 可延迟加载的 ``模块:函数`` 引用。

    配置文件导入时不立即导入整个任务实现；环境真正创建 manager 时才解析该函数，
    从而避免配置模块和任务模块互相导入形成循环。
    """

    return f"{_TASK_MODULE}:{name}"


URDF_PATH = str(Path(__file__).resolve().parents[1] / "data/my_robot/urdf/my_exoskeleton.urdf")
# 第一课程阶段的固定前进目标 [m/s]。速度使用机器人自身坐标系，+X 为正前方。
FIXED_FORWARD_SPEED_M_S = 0.2
# Isaac Lab 的课程边界单位是“全局环境控制步”，不是 PPO epoch。默认每个 PPO rollout
# 含 32 个控制步，所以 500 个 PPO epoch = 500 * 32 = 16,000 个控制步。阶段二从
# 16,000 步开始，阶段三从 32,000 步开始。断点续训时环境步数不会随 RL-Games
# checkpoint 一起保存，需要给课程项设置 step_offset 才能接着原阶段训练。
CURRICULUM_ITERATIONS_PER_PHASE = 500
PPO_HORIZON_LENGTH = 32
PHASE_2_START_STEP = CURRICULUM_ITERATIONS_PER_PHASE * PPO_HORIZON_LENGTH
PHASE_3_START_STEP = 2 * CURRICULUM_ITERATIONS_PER_PHASE * PPO_HORIZON_LENGTH
MAX_CURRICULUM_SPEED_M_S = 4.0


@configclass
class MyExoskeletonSceneCfg(InteractiveSceneCfg):
    """场景层：描述每个并行环境中的地面、机器人、传感器和灯光。"""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=URDF_PATH,
            fix_base=False,
            merge_fixed_joints=True,
            make_instanceable=False,
            activate_contact_sensors=True,
            # 按任务要求保持 self-collision 开启。导入的 STL 凸包在标称关节姿态下存在
            # 重叠，因此还需要专用低面数碰撞网格和相邻连杆过滤，才能把视觉结果视为
            # 最终模型。
            self_collision=True,
            # URDF 导入器设置改变时重新生成缓存 USD。若没有此标志，即使下面的运行时
            # articulation 属性要求 self-collision，旧 payload 仍可能保留关闭状态。
            force_usd_conversion=True,
            # 完整 CAD 三角网格过于凹陷且噪声较多，无法稳定计算自身接触。凸分解能在
            # 保持 self-collision 的同时，避免巨大穿透冲量在渲染器中表现为连杆变形。
            collision_type="Convex Decomposition",
            merge_mesh=False,
            # Isaac Lab 3.0 把 PhysX 专用 schema 配置移到了
            # ``isaaclab_physx.sim.schemas``。这里直接使用具体类，让 self-collision 和
            # 求解器字段直接写入 ``physxArticulation`` 命名空间，不经过转发别名。
            articulation_props=PhysxArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=2
            ),
        ),
        # URDF 关节限位已经包含少量人体伸展余量，因此软限位系数保持 1.0。左右膝轴
        # 是镜像且限位不对称，若围绕中点再次缩小范围，0 rad 伸直姿态可能反而越界。
        soft_joint_pos_limit_factor=1.0,
        # URDF 中脚底已经接近根坐标系 z=0，所以根节点只放到地面上方 1 cm。若初始
        # 高度写成普通人身高，模型会在每次 reset 时从空中掉落，而不是从站姿开始。
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)),
        actuators={
            "exoskeleton_pd": ImplicitActuatorCfg(
                joint_names_expr=list(ACTUATED_JOINT_NAMES),
                # 隐式 PD 控制器把后面的“目标角度”转换为仿真力矩。stiffness 类似
                # 弹簧强度，damping 抑制快速振荡；力矩和速度上限均按关节执行。
                stiffness=80.0,
                damping=4.0,
                joint_effort_limit={name: 60.0 for name in ACTUATED_JOINT_NAMES},
                joint_velocity_limit={name: 3.14 for name in ACTUATED_JOINT_NAMES},
            )
        },
    )

    # URDF 中的 IMU 安装在这些连杆上。由于模型没有骨盆/躯干 IMU，这里用物理意义上
    # 等价的根节点状态估计基座状态。
    imu_rthigh = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/R_turn/Rthigh", class_type=_INSPECTABLE_IMU_CLASS
    )
    imu_rleg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/R_turn/Rthigh/Rleg", class_type=_INSPECTABLE_IMU_CLASS
    )
    imu_rfoot = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/R_turn/Rthigh/Rleg/Rfoot", class_type=_INSPECTABLE_IMU_CLASS
    )
    imu_lthigh = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/Lturn/Lthigh", class_type=_INSPECTABLE_IMU_CLASS
    )
    imu_lleg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/Lturn/Lthigh/Lleg", class_type=_INSPECTABLE_IMU_CLASS
    )
    imu_lfoot = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/Lturn/Lthigh/Lleg/Lfoot", class_type=_INSPECTABLE_IMU_CLASS
    )

    left_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/Lturn/Lthigh/Lleg/Lfoot",
        class_type=_NESTED_CONTACT_SENSOR_CLASS,
        history_length=3,
        # 策略只需净接触力；若请求接触点坐标，PhysX 还要求显式配置过滤集合。
        track_contact_points=False,
        track_air_time=True,
    )
    right_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/R_turn/Rthigh/Rleg/Rfoot",
        class_type=_NESTED_CONTACT_SENSOR_CLASS,
        history_length=3,
        track_contact_points=False,
        track_air_time=True,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.78, 0.82)),
    )


@configclass
class MyExoskeletonCommandsCfg:
    """命令层：生成策略应该追踪的机身速度和当前步态标签。"""

    @configclass
    class BaseVelocityCfg(mdp.UniformVelocityCommandCfg):
        """基础速度命令，以及可选的单轨迹传感器参考回放设置。"""

        class_type = _task_term("GaitVelocityCommand")
        reference_data_path: str | None = None
        reference_target_dt_s: float = CONTROL_DT_S
        reference_loop: bool = True
        reference_random_start: bool = False

    base_velocity = BaseVelocityCfg(
        class_type=_task_term("GaitVelocityCommand"),
        asset_name="robot",
        # 每次随机速度命令保持 3~6 s，才能完成 1.5 s 行走转站立和 2.5 s 跑步转
        # 行走判定。命令保持时间与步态状态机更新率不同：状态机每 0.1 s 更新一次。
        resampling_time_range=(3.0, 6.0),
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=False,
        # 上游默认误差阈值过宽：目标仅为 0.2 m/s 时，静止也可能被判断为成功。
        # 这里收紧阈值，避免策略通过“一直站着”绕开前进任务。
        vel_xy_success_threshold=0.05,
        vel_yaw_success_threshold=0.10,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(FIXED_FORWARD_SPEED_M_S, FIXED_FORWARD_SPEED_M_S),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class MyExoskeletonObservationsCfg:
    """观测层：定义 actor 的 49 维输入和 critic 的 62 维特权输入。"""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """动作网络输入，函数返回形状 ``[N, 49]`` 的连续特征。"""

        state = ObservationTermCfg(func=_task_term("exoskeleton_observation"))

        def __post_init__(self):
            """把组内 term 拼成 ``[N, 49]``，并关闭管理器额外施加的随机观测扰动。"""

            # 一个组内可能有多个 ObservationTerm；开启拼接后最后一维会合并成一个
            # 普通张量。此任务关闭随机观测破坏，传感器开关由环境显式处理。
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """价值网络输入：49 维策略状态加仿真专有接触/动力学量，共 ``[N, 62]``。"""

        state = ObservationTermCfg(func=_task_term("exoskeleton_critic_observation"))

        def __post_init__(self):
            """把 critic 的特权 term 拼成 ``[N, 62]``，并保持观测值不被自动 corruption。"""

            self.concatenate_terms = True
            self.enable_corruption = False

    critic: CriticCfg = CriticCfg()


@configclass
class MyExoskeletonActionsCfg:
    """动作层：把 actor 的 ``[N, 8]`` 归一化动作变成 8 个绝对位置目标。

    ``use_default_offset=True`` 时计算关系是
    ``目标角 = 默认关节角 + 网络动作 * JOINT_ACTION_SCALE_RAD``。这里的“相对默认
    姿态”不能理解成“相对当前角度的增量”：同一个动作连续发送两次，目标角不会累加
    两次。随后隐式 PD 执行器负责追踪这个目标角。
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ACTUATED_JOINT_NAMES),
        scale=dict(JOINT_ACTION_SCALE_RAD),
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class MyExoskeletonRewardsCfg:
    """奖励层：公共运动奖励和按步态选择生效的奖励。

    ``weight`` 为正时鼓励该行为，为负时表示惩罚。Isaac Lab 的 RewardManager 每个
    控制周期调用这些函数并汇总。函数返回“原始分数”，权重在 manager 中统一应用；
    ``walk_*``、``stand_*`` 等项还会乘步态掩码，避免不同任务互相冲突。
    """

    # 该模型只能感知两只脚，因此保留终止惩罚，作为原论文碰撞惩罚的可行替代。
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
    alive = RewardTermCfg(func=mdp.is_alive, weight=0.15)
    track_linear_velocity = RewardTermCfg(
        func=_task_term("tracking_linear_velocity_reward"),
        weight=1.5,
        params={"std": 0.5, "activity_gate_speed": 0.1, "activity_gate_command_max": 0.25},
    )
    # 第一阶段若只使用论文中较宽松的指数跟踪项，静止机器人也能得到
    # exp(-0.2^2 / 0.5^2) = 0.852。这个辅助窗口在静止时为零，在引入后续命令范围前
    # 恢复学会第一个行走周期所需的探索压力。
    fixed_forward_walk = RewardTermCfg(
        func=_task_term("fixed_forward_walk_reward"),
        weight=3.0,
        params={
            "target_speed": FIXED_FORWARD_SPEED_M_S,
            "speed_tolerance": FIXED_FORWARD_SPEED_M_S,
            "lateral_speed_std": 0.15,
            "yaw_rate_std": 0.5,
            "command_speed_tolerance": 0.1,
            "gait": 1,
        },
    )
    # 未配置参考路径时实际等同于禁用：任务项会返回全零张量。针对特定记录重新训练时，
    # 需要设置 ``reference_data_path`` 并调整此权重。
    sensor_reference_imitation = RewardTermCfg(
        func=_task_term("sensor_reference_imitation_reward"),
        weight=1.0,
        params={
            "joint_pos_weight": 1.0,
            "joint_vel_weight": 0.25,
            "gyro_weight": 0.5,
            "contact_weight": 0.25,
            "joint_pos_sigma": 0.35,
            "joint_vel_sigma": 1.5,
            "gyro_sigma": 1.0,
            "contact_sigma": 0.35,
        },
    )
    track_yaw_velocity = RewardTermCfg(func=_task_term("tracking_yaw_velocity_reward"), weight=1.0, params={"std": 0.5})
    vertical_linear_velocity = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
    horizontal_angular_velocity = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation_deviation = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewardTermCfg(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.01})
    joint_acceleration = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_velocity = RewardTermCfg(func=mdp.joint_vel_l2, weight=-1.0e-3)
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    joint_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0)
    knee_human_range = RewardTermCfg(
        func=_task_term("knee_limit_penalty"),
        weight=-12.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(KNEE_JOINT_NAMES), preserve_order=True),
            "bounds_by_joint": dict(KNEE_ARTICULATION_LIMITS_RAD),
        },
    )
    knee_velocity_limit = RewardTermCfg(
        func=_task_term("knee_velocity_limit_penalty"),
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(KNEE_JOINT_NAMES), preserve_order=True),
            "max_velocity": 3.14,
        },
    )
    hip_position_deviation = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Rturn_to_Rthigh", "Lturn_to_Lthigh"],
                preserve_order=True,
            )
        },
    )
    leg_momentum_balance = RewardTermCfg(
        func=_task_term("leg_momentum_balance_reward"),
        weight=5.0,
        params={
            # 形态自适应的两侧动量项要求刚体按“右腿在前、左腿在后”排列。
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "R_turn",
                    "Rthigh",
                    "Rleg",
                    "Rfoot",
                    "Lturn",
                    "Lthigh",
                    "Lleg",
                    "Lfoot",
                ],
                preserve_order=True,
            ),
            "momentum_scale": 10.0,
        },
    )
    leg_mechanical_power = RewardTermCfg(
        func=_task_term("leg_mechanical_power_penalty"),
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTUATED_JOINT_NAMES), preserve_order=True),
            "power_scale": 50.0,
        },
    )
    leg_phase_symmetry = RewardTermCfg(
        func=_task_term("leg_phase_symmetry_reward"),
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "base_to_Rturn",
                    "Rturn_to_Rthigh",
                    "Rthigh_to_Rleg",
                    "Rleg_to_Rfoot",
                    "base_to_Lturn",
                    "Lturn_to_Lthigh",
                    "Lthigh_to_Lleg",
                    "Lleg_to_Lfoot",
                ],
                preserve_order=True,
            ),
            "sigma": 0.5,
            "activity_scale": 0.5,
            "motion_scale": FIXED_FORWARD_SPEED_M_S,
        },
    )
    leg_amplitude_match = RewardTermCfg(
        func=_task_term("leg_amplitude_match_reward"),
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "base_to_Rturn",
                    "Rturn_to_Rthigh",
                    "Rthigh_to_Rleg",
                    "Rleg_to_Rfoot",
                    "base_to_Lturn",
                    "Lturn_to_Lthigh",
                    "Lthigh_to_Lleg",
                    "Lleg_to_Lfoot",
                ],
                preserve_order=True,
            ),
            "sigma": 0.5,
            "activity_scale": 0.5,
            "motion_scale": FIXED_FORWARD_SPEED_M_S,
        },
    )

    # 行走奖励。
    walk_swing_height = RewardTermCfg(
        func=_task_term("swing_height_penalty"),
        weight=-15.0,
        params={
            "gait": 1,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Lfoot", "Rfoot"], preserve_order=True),
            "target_height": 0.08,
        },
    )
    walk_contact = RewardTermCfg(func=_task_term("phase_contact_reward"), weight=1.0, params={"gait": 1})
    walk_straight_knee = RewardTermCfg(
        func=_task_term("straight_knee_reward"),
        weight=0.1,
        params={
            "gait": 1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Rthigh_to_Rleg", "Lthigh_to_Lleg"], preserve_order=True),
            "sigma": 0.35,
        },
    )
    walk_human_knee = RewardTermCfg(
        func=_task_term("human_knee_gait_reward"),
        weight=1.5,
        params={
            "gait": 1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(KNEE_JOINT_NAMES), preserve_order=True),
            "stance_target": 0.0,
            "swing_target": 0.8,
            "stance_sigma": 0.30,
            "swing_sigma": 0.45,
        },
    )
    walk_foot_drag = RewardTermCfg(
        func=_task_term("foot_drag_penalty"),
        weight=-0.5,
        params={
            "gait": 1,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Lfoot", "Rfoot"], preserve_order=True),
        },
    )

    # 站立奖励。
    stand_contact = RewardTermCfg(func=_task_term("double_support_reward"), weight=2.5)
    stand_base_motion = RewardTermCfg(
        func=_task_term("standing_base_stillness_reward"), weight=2.5, params={"sigma": 0.2}
    )
    stand_pose = RewardTermCfg(func=_task_term("standing_pose_reward"), weight=4.0, params={"sigma": 0.35})
    stand_feet_alignment = RewardTermCfg(
        func=_task_term("standing_feet_alignment_reward"),
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Lfoot", "Rfoot"], preserve_order=True),
            "sigma": 0.2,
        },
    )
    stand_uprightness = RewardTermCfg(func=_task_term("standing_upright_reward"), weight=1.0, params={"sigma": 0.2})
    stand_foot_flatness = RewardTermCfg(
        func=_task_term("standing_foot_flatness_reward"),
        weight=2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Lfoot", "Rfoot"], preserve_order=True),
            "sigma": 0.2,
        },
    )
    stand_stillness = RewardTermCfg(
        func=_task_term("standing_joint_stillness_reward"), weight=2.0, params={"sigma": 0.25}
    )
    stand_foot_drag = RewardTermCfg(
        func=_task_term("foot_drag_penalty"),
        weight=-0.2,
        params={
            "gait": 0,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Lfoot", "Rfoot"], preserve_order=True),
        },
    )

    # 行走转站立奖励。
    w2s_swing_height = RewardTermCfg(
        func=_task_term("swing_height_penalty"),
        weight=-20.0,
        params={
            "gait": 2,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Lfoot", "Rfoot"], preserve_order=True),
            "target_height": 0.06,
        },
    )
    w2s_contact = RewardTermCfg(
        func=_task_term("phase_contact_reward"),
        weight=1.0,
        params={"gait": 2, "allow_double_support": True, "double_support_ramp_s": 1.5},
    )
    w2s_smooth_slowdown = RewardTermCfg(
        func=_task_term("smooth_slowdown_reward"),
        weight=0.1,
        params={"gait": 2, "scale": 0.2},
    )
    w2s_foot_drag = RewardTermCfg(
        func=_task_term("foot_drag_penalty"),
        weight=-0.1,
        params={
            "gait": 2,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Lfoot", "Rfoot"], preserve_order=True),
        },
    )

    # 跑步奖励。
    run_contact = RewardTermCfg(
        func=_task_term("phase_contact_reward"),
        weight=1.0,
        params={"gait": 3, "flight_window": 0.35},
    )
    run_foot_drag = RewardTermCfg(
        func=_task_term("foot_drag_penalty"),
        weight=-1.0,
        params={
            "gait": 3,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Lfoot", "Rfoot"], preserve_order=True),
        },
    )
    run_short_contact = RewardTermCfg(
        func=_task_term("short_contact_reward"), weight=0.2, params={"max_contact_time": 0.25}
    )
    run_swing_height = RewardTermCfg(
        func=_task_term("swing_height_penalty"),
        weight=-25.0,
        params={
            "gait": 3,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Lfoot", "Rfoot"], preserve_order=True),
            "target_height": 0.12,
        },
    )
    run_push_off = RewardTermCfg(
        func=_task_term("push_off_reward"),
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Rleg_to_Rfoot", "Lleg_to_Lfoot"], preserve_order=True)
        },
    )
    run_forward_velocity = RewardTermCfg(
        func=_task_term("forward_velocity_reward"),
        weight=0.2,
        params={"gait": 3, "scale": MAX_CURRICULUM_SPEED_M_S},
    )

    # 跑步转行走奖励。
    r2w_contact = RewardTermCfg(func=_task_term("phase_contact_reward"), weight=1.0, params={"gait": 4})
    r2w_smooth_slowdown = RewardTermCfg(
        func=_task_term("smooth_slowdown_reward"),
        weight=1.0,
        params={"gait": 4, "scale": 0.5},
    )
    r2w_walk_speed = RewardTermCfg(
        func=_task_term("transition_to_walk_speed_reward"),
        weight=0.5,
        params={"target_speed": FIXED_FORWARD_SPEED_M_S, "std": 0.3},
    )
    r2w_swing_height = RewardTermCfg(
        func=_task_term("swing_height_penalty"),
        weight=-25.0,
        params={
            "gait": 4,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Lfoot", "Rfoot"], preserve_order=True),
            "target_height": 0.08,
        },
    )


@configclass
class MyExoskeletonEventsCfg:
    """事件层：物理参数随机化、episode 重置和训练中的外部扰动。

    初学者需要特别区分三个 ``mode``：

    * ``startup``：创建环境时执行一次，不会在每个 episode reset 时重新采样；
    * ``reset``：某个并行环境结束并重置时执行；
    * ``interval``：仿真运行期间按给定秒数周期执行。

    因而当前摩擦、质量和 PD 增益是在一次训练进程启动时随机化，而不是每回合随机化。
    4096 个并行环境可以在启动时得到不同参数，但同一个环境在后续 reset 后会沿用它的
    启动参数。如需“每个 episode 重采样”，必须另行把对应项设计为 reset 事件并验证
    Isaac Lab API 支持，不能仅修改这段说明。
    """

    randomize_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # 静摩擦决定“开始滑动有多难”，动摩擦决定“已经滑动后阻力多大”。
            # make_consistent=True 会确保采样出的材料参数保持物理上一致。
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    randomize_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )
    randomize_actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTUATED_JOINT_NAMES)),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    reset_scene_to_default = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    lateral_push = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        # 每 3~6 秒给机身一次随机侧向/偏航速度扰动，用来训练受扰后的恢复能力。
        interval_range_s=(3.0, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (0.0, 0.0), "y": (-0.3, 0.3), "yaw": (-0.2, 0.2)},
        },
    )


@configclass
class MyExoskeletonCurriculumCfg:
    """课程层：按全局控制步数从固定行走逐渐扩展到转换和多步态。"""

    gait_commands = CurriculumTermCfg(
        func=_task_term("advance_exoskeleton_curriculum"),
        params={
            "phase2_start": PHASE_2_START_STEP,
            "phase3_start": PHASE_3_START_STEP,
            # 断点续训时应设为“旧 PPO epoch 数 * horizon_length”。RL-Games checkpoint
            # 保存网络/优化器状态，却不保存环境的 common_step_counter；保持 0 会让
            # 恢复训练后的环境课程重新从第一阶段开始。
            "step_offset": 0,
            "walk_speed": FIXED_FORWARD_SPEED_M_S,
            "transition_max_speed": 0.8,
            "multi_gait_max_speed": MAX_CURRICULUM_SPEED_M_S,
        },
    )


@configclass
class MyExoskeletonTerminationsCfg:
    """终止层：任一条件成立时结束对应环境的本轮 episode 并立即 reset。

    ``time_out`` 是正常达到 20 s 上限；``fallen`` 检查机身倾斜是否超过 65 度；膝关节
    越过人体范围也会终止。终止并不代表整个训练进程停止，只重置触发条件的并行环境。
    """

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    fallen = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.radians(65.0)},
    )
    knee_out_of_human_range = TerminationTermCfg(
        func=_task_term("knee_out_of_human_range"),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(KNEE_JOINT_NAMES), preserve_order=True),
            "bounds_by_joint": dict(KNEE_ARTICULATION_LIMITS_RAD),
        },
    )


@configclass
class MyExoskeletonFlatEnvCfg(ManagerBasedRLEnvCfg):
    """总装层：把场景、观测、动作、命令、奖励和生命周期管理器组成一个 RL 环境。"""

    scene: MyExoskeletonSceneCfg = MyExoskeletonSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: MyExoskeletonObservationsCfg = MyExoskeletonObservationsCfg()
    actions: MyExoskeletonActionsCfg = MyExoskeletonActionsCfg()
    commands: MyExoskeletonCommandsCfg = MyExoskeletonCommandsCfg()
    rewards: MyExoskeletonRewardsCfg = MyExoskeletonRewardsCfg()
    terminations: MyExoskeletonTerminationsCfg = MyExoskeletonTerminationsCfg()
    # 即使没有启用任何 term，各 manager 仍要求提供配置对象。
    events: MyExoskeletonEventsCfg = MyExoskeletonEventsCfg()
    curriculum: MyExoskeletonCurriculumCfg = MyExoskeletonCurriculumCfg()
    # 单个 episode 最长 20 s；50 Hz 控制率下最多约 1000 个策略步。
    episode_length_s = 20.0
    # 可选传感器开关与模仿子配置共用，使便捷 CLI 对旧版平地任务仍然有效。平地场景没有
    # RGB/Event 实体，因此对应标志只作为无操作的兼容开关；IMU/contact 项则会遵守它们。
    use_imu: bool = True
    use_rgb: bool = True
    use_event: bool = True
    use_contact: bool = True
    # 可选的播放结果导出。actor 始终只输出 8 个动作；记录器读取 8 个目标角，并按
    # 相邻采样及其真实/配置时间差求出 8 个速度和 8 个角加速度，合成 24 维结果。
    # 这不会改变旧 checkpoint 的网络输出层，也不代表模型直接预测了速度/加速度。
    ruleresult_enabled: bool = False
    ruleresult_interval: int = 1
    ruleresult_env_id: int = 0
    ruleresult_output_path: str | None = None
    ruleresult_output_format: str = "jsonl"
    ruleresult_protocol_config_path: str | None = None
    # ruleresult_dt_s 是标称周期和异常时间戳时的兜底值。simulation 使用仿真时间，
    # wall_clock 使用工控机单调时钟，configured 则始终假设固定周期；差分计算会把
    # 实际采用的 dt_source 一并写入 JSONL，便于发现输出卡顿。
    ruleresult_dt_s: float = CONTROL_DT_S
    ruleresult_time_mode: str = "simulation"
    ruleresult_min_dt_s: float = 1.0e-6
    ruleresult_max_dt_s: float = 10.0
    ruleresult_prediction_horizon: int = 0
    ruleresult_prediction_mode: str = "none"
    ruleresult_prediction_dt_s: float | None = None

    def __post_init__(self):
        """在各层配置组装后，统一设置时间步、空间维度并检查导出参数。"""

        # 每个策略动作保持 4 个物理小步：0.005 s * 4 = 0.02 s，即标称 50 Hz。
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # Gym 空间这里只声明最后一维大小；实际批量张量分别为 [N, 8]、[N, 49]
        # 和 [N, 62]，N 由 scene.num_envs 或命令行 --num_envs 决定。
        self.action_space = len(ACTUATED_JOINT_NAMES)
        self.observation_space = POLICY_OBS_DIM
        self.state_space = CRITIC_OBS_DIM
        if self.ruleresult_interval <= 0:
            raise ValueError("ruleresult_interval must be positive")
        if self.ruleresult_env_id < 0:
            raise ValueError("ruleresult_env_id must be non-negative")
        output_format = str(self.ruleresult_output_format).strip().lower()
        output_format = {"json": "jsonl", "55aa": "protocol_55aa", "protocol-55aa": "protocol_55aa"}.get(
            output_format, output_format
        )
        if output_format not in {"jsonl", "protocol_55aa"}:
            raise ValueError("ruleresult_output_format must be jsonl or protocol_55aa")
        self.ruleresult_output_format = output_format
        if output_format == "protocol_55aa" and (
            self.ruleresult_protocol_config_path is None or not str(self.ruleresult_protocol_config_path).strip()
        ):
            raise ValueError("ruleresult_protocol_config_path is required for protocol_55aa output")
        if not math.isfinite(float(self.ruleresult_dt_s)) or self.ruleresult_dt_s <= 0.0:
            raise ValueError("ruleresult_dt_s must be positive and finite")
        time_mode = str(self.ruleresult_time_mode).strip().lower()
        if time_mode in {"sim", "wall", "actual"}:
            time_mode = {"sim": "simulation", "wall": "wall_clock", "actual": "wall_clock"}[time_mode]
            self.ruleresult_time_mode = time_mode
        if time_mode not in {"simulation", "wall_clock", "configured"}:
            raise ValueError("ruleresult_time_mode must be simulation, wall_clock, or configured")
        if not math.isfinite(float(self.ruleresult_min_dt_s)) or self.ruleresult_min_dt_s <= 0.0:
            raise ValueError("ruleresult_min_dt_s must be positive and finite")
        if not math.isfinite(float(self.ruleresult_max_dt_s)) or self.ruleresult_max_dt_s < self.ruleresult_min_dt_s:
            raise ValueError("ruleresult_max_dt_s must be finite and >= ruleresult_min_dt_s")
        if self.ruleresult_prediction_horizon < 0:
            raise ValueError("ruleresult_prediction_horizon must be non-negative")
        prediction_mode = str(self.ruleresult_prediction_mode).strip().lower()
        if prediction_mode in {"constant_acceleration", "extrapolate"}:
            prediction_mode = "kinematic"
            self.ruleresult_prediction_mode = prediction_mode
        if prediction_mode not in {"none", "kinematic", "supplied"}:
            raise ValueError("ruleresult_prediction_mode must be none, kinematic, or supplied")
        if self.ruleresult_prediction_dt_s is not None and (
            not math.isfinite(float(self.ruleresult_prediction_dt_s)) or self.ruleresult_prediction_dt_s <= 0.0
        ):
            raise ValueError("ruleresult_prediction_dt_s must be positive and finite when provided")


# 任务规格要求保留的向后兼容名称。
MyExoskeletonEnvCfg = MyExoskeletonFlatEnvCfg
