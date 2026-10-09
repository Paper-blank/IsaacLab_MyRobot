"""在基础平地运动环境上增加多模态动作模仿的配置。

这个文件不是从零创建第二套机器人，而是通过 Python 继承复用
:mod:`my_exoskeleton_env_cfg`：基础场景、8 关节动作、行走/站立奖励、终止条件、课程
学习和域随机化仍然生效；这里只替换或扩展模仿任务特有的场景、命令、观测和奖励。

额外输入包括 6 个映射后的 IMU、2 台固定 RGB 相机和 2 台事件相机。原始图像不会直接
作为数十万维 PPO 输入，而会压缩成固定长度特征。默认 actor 输入为
``49 基础状态 + 6*9 IMU + 12 RGB + 32 事件 = 147`` 维，critic 仍使用 62 维仿真
特权状态，actor 输出仍为 8 个关节动作。这样才能继续使用普通 RL-Games MLP/LSTM，
并在大量并行环境中控制显存占用。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.envs.my_exoskeleton_env_cfg import (
    FIXED_FORWARD_SPEED_M_S,
    MyExoskeletonCommandsCfg,
    MyExoskeletonFlatEnvCfg,
    MyExoskeletonRewardsCfg,
    MyExoskeletonSceneCfg,
)
from isaaclab_assets.tasks.imitation_reward import EVENT_FEATURES, IMU_COUNT, IMU_FEATURES, RGB_FEATURES
from isaaclab_assets.tasks.robot_contract import ACTUATED_JOINT_NAMES, CRITIC_OBS_DIM, POLICY_OBS_DIM
from isaaclab_assets.tasks.vision_sensor_cfg import (
    CAMERA_X_M,
    CHANNEL_LENGTH_M,
    DEFAULT_CHANNEL_HALF_WIDTH_M,
    event_camera_plugin_available,
    make_deferred_event_camera_cfg,
    make_event_camera_cfg,
    make_rgb_camera_cfg,
)

_TASK_MODULE = "isaaclab_assets.tasks.imitation_reward"
_FLAT_TASK_MODULE = "isaaclab_assets.tasks.my_exoskeleton_flat"


def _term(module: str, name: str) -> str:
    """生成延迟加载的 ``模块:函数`` 引用，避免配置导入时启动传感器后端。"""

    return f"{module}:{name}"


_INSPECTABLE_IMU_CLASS = "isaaclab_assets.sensors.inspectable_imu:InspectableImu"


def _event_cfg(name: str, side: str):
    """已安装 EVIS 时创建真实事件相机，否则创建会明确标记无效的占位配置。

    占位配置只为保持配置结构和观测形状稳定，不会伪造事件；对应奖励会被有效性掩码
    置零。
    """

    if event_camera_plugin_available():
        return make_event_camera_cfg(name, side)  # type: ignore[arg-type]
    return make_deferred_event_camera_cfg(name, side)  # type: ignore[arg-type]


@configclass
class MyExoskeletonImitationSceneCfg(MyExoskeletonSceneCfg):
    """场景扩展：2.4 m 可视通道、双目 RGB/DVS 相机和 6 个数据集同名 IMU。"""

    # 特意保留 ``TerrainImporterCfg(terrain_type="plane")`` 作为物理地面。但它无限大且
    # 没有可见的通道边界，Viser/Newton 只能显示空白平面。这些轻量、仅用于显示的长方体
    # 为交互查看器和 RGB 相机提供明确的局部通道轮廓；它们随每个环境克隆，不增加接触
    # 或动力学效果。
    channel_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Channel/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(CHANNEL_LENGTH_M, 2.0 * DEFAULT_CHANNEL_HALF_WIDTH_M, 0.012),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.20, 0.24), roughness=0.92),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(CAMERA_X_M, 0.0, 0.006)),
    )
    channel_left_edge = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Channel/LeftEdge",
        spawn=sim_utils.CuboidCfg(
            size=(CHANNEL_LENGTH_M, 0.045, 0.045),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.66, 0.12), roughness=0.75),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(CAMERA_X_M, DEFAULT_CHANNEL_HALF_WIDTH_M + 0.025, 0.0225)),
    )
    channel_right_edge = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Channel/RightEdge",
        spawn=sim_utils.CuboidCfg(
            size=(CHANNEL_LENGTH_M, 0.045, 0.045),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.66, 0.12), roughness=0.75),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(CAMERA_X_M, -(DEFAULT_CHANNEL_HALF_WIDTH_M + 0.025), 0.0225)),
    )

    # 父类使用按身体部位命名的 IMU；模仿数据使用 imu_01..imu_06。先清除父类字段，
    # 再按记录仪编号创建，防止同一连杆重复挂载两套传感器。
    imu_rthigh = None
    imu_rleg = None
    imu_rfoot = None
    imu_lthigh = None
    imu_lleg = None
    imu_lfoot = None

    # 记录仪顺序按左右交错：01/02 为左/右踝，03/04 为左/右膝，05/06 为左/右髋。
    # 运行时一个 IMU 张量形状为 [N, 6, 9]，名称与记录仪 ID 相同可避免传感器下标歧义。
    imu_01 = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/Lturn/Lthigh/Lleg/Lfoot",
        class_type=_INSPECTABLE_IMU_CLASS,
        update_period=0.02,
    )
    imu_02 = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/R_turn/Rthigh/Rleg/Rfoot",
        class_type=_INSPECTABLE_IMU_CLASS,
        update_period=0.02,
    )
    imu_03 = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/Lturn/Lthigh/Lleg",
        class_type=_INSPECTABLE_IMU_CLASS,
        update_period=0.02,
    )
    imu_04 = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/R_turn/Rthigh/Rleg",
        class_type=_INSPECTABLE_IMU_CLASS,
        update_period=0.02,
    )
    imu_05 = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/Lturn/Lthigh",
        class_type=_INSPECTABLE_IMU_CLASS,
        update_period=0.02,
    )
    imu_06 = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/R_turn/Rthigh",
        class_type=_INSPECTABLE_IMU_CLASS,
        update_period=0.02,
    )

    left_rgb_camera = make_rgb_camera_cfg("left_rgb_camera", "left")
    right_rgb_camera = make_rgb_camera_cfg("right_rgb_camera", "right")
    # EVIS 使用这些精确名称查找异步事件流。回退适配器保留名称和位姿，但会把事件流
    # 标记为无效。
    left_event_camera = _event_cfg("left_event_camera", "left")
    right_event_camera = _event_cfg("right_event_camera", "right")

    # 元数据属性特意不声明为配置字段。场景收集器会把每个额外字段都当成实体，否则就会
    # 尝试把这些浮点数展开成 USD prim 路径。
    @property
    def channel_length_m(self) -> float:
        """返回可视通道的全长（米）；属性不会被场景收集器误当成新的 USD 实体。"""

        return CHANNEL_LENGTH_M

    @property
    def channel_half_width_m(self) -> float:
        """返回通道中心线到单侧边界的距离（米），供配置检查和相机几何共用。"""

        return DEFAULT_CHANNEL_HALF_WIDTH_M


@configclass
class MyExoskeletonImitationCommandsCfg(MyExoskeletonCommandsCfg):
    """命令扩展：记录选中的 BT/AT 任务，并支持在并行环境间随机抽取任务。"""

    @configclass
    class BaseVelocityCfg(MyExoskeletonCommandsCfg.BaseVelocityCfg):
        """在基础速度命令上增加数据集任务选择和连续教师窗口长度。"""

        # task_id=all 表示由数据集提供的任务中选择；window_size 是一次保留的连续
        # 50 Hz 教师帧数量，而不是并行环境数。
        task_id: str = "all"
        random_task_sampling: bool = True
        window_size: int = 1

    base_velocity: BaseVelocityCfg = BaseVelocityCfg(
        class_type=_term("isaaclab_assets.tasks.my_exoskeleton_flat", "GaitVelocityCommand"),
        asset_name="robot",
        resampling_time_range=(3.0, 6.0),
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=False,
        vel_xy_success_threshold=0.05,
        vel_yaw_success_threshold=0.10,
        ranges=MyExoskeletonCommandsCfg.BaseVelocityCfg.Ranges(
            lin_vel_x=(FIXED_FORWARD_SPEED_M_S, FIXED_FORWARD_SPEED_M_S),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class MyExoskeletonImitationObservationsCfg:
    """多模态观测层：把不同来源保留为独立组，再由 RL-Games 拼接。

    PPO YAML 中 ``obs: [policy, imu, camera_rgb, event_camera]`` 决定 actor 使用哪些组。
    每组内部的 ``concatenate_terms=True`` 只负责把该组的 term 合成普通二维批量张量。
    禁用某个传感器时函数仍返回相同形状的零张量，因此 checkpoint 的输入维度不变。
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """actor 的基础状态组，输出形状为 ``[N, 49]``。"""

        # 基础本体感觉、接触、命令和步态信息，形状 [N, 49]。
        state = ObservationTermCfg(func=_term(_FLAT_TASK_MODULE, "exoskeleton_observation"))

        def __post_init__(self):
            """将组内 term 沿末维拼接，并关闭管理器自动添加的观测 corruption。"""

            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class ImuCfg(ObservationGroupCfg):
        """6 个映射后 IMU 的观测组，展平输出形状为 ``[N, 54]``。"""

        # 6 个 IMU *（姿态3 + 角速度3 + 线加速度3），展平后形状 [N, 54]。
        imu = ObservationTermCfg(func=_term(_TASK_MODULE, "imu_observation"))

        def __post_init__(self):
            """将 IMU term 拼成二维批量张量，并避免在显式传感器噪声外重复 corruption。"""

            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CameraRgbCfg(ObservationGroupCfg):
        """左右 RGB 相机的紧凑统计特征组，输出形状为 ``[N, 12]``。"""

        # 左右相机各自 RGB 均值和标准差，共 12 个紧凑特征，形状 [N, 12]。
        camera_rgb = ObservationTermCfg(func=_term(_TASK_MODULE, "camera_rgb_observation"))

        def __post_init__(self):
            """将 RGB 特征 term 拼接成一个张量，并关闭额外的管理器级 corruption。"""

            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class EventCameraCfg(ObservationGroupCfg):
        """双事件相机的极性直方图特征组，输出形状为 ``[N, 32]``。"""

        # 2 相机 * 2 极性 * 8 空间池化格，共 32 个特征，形状 [N, 32]。
        event_camera = ObservationTermCfg(func=_term(_TASK_MODULE, "event_camera_observation"))

        def __post_init__(self):
            """将事件特征 term 拼接为固定 shape，并关闭可能改变有效性语义的 corruption。"""

            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """仅供训练价值网络使用的特权状态组，输出形状为 ``[N, 62]``。"""

        # critic 的仿真特权状态，形状 [N, 62]；不会成为实体机 actor 的输入要求。
        state = ObservationTermCfg(func=_term(_FLAT_TASK_MODULE, "exoskeleton_critic_observation"))

        def __post_init__(self):
            """将 critic term 拼接成二维张量，并保持仿真特权值不受自动 corruption。"""

            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    imu: ImuCfg = ImuCfg()
    camera_rgb: CameraRgbCfg = CameraRgbCfg()
    event_camera: EventCameraCfg = EventCameraCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class MyExoskeletonImitationRewardsCfg(MyExoskeletonRewardsCfg):
    """奖励扩展：在父类全部运动/安全奖励之上，再叠加 6 个正相似度奖励。

    各相似度大致采用 ``exp(-均方误差 / sigma^2)``，越接近数据集目标越接近 1；缺失
    或已禁用的传感器通过有效性掩码贡献 0。这里列出的 ``weight`` 才是训练实际使用的
    manager 权重；``configs/imitation_reward.yaml`` 是便于阅读的契约说明，目前不会自动
    覆盖这些 Python 数值。
    """

    imitation_imu = RewardTermCfg(
        func=_term(_TASK_MODULE, "imitation_imu_reward"),
        weight=1.0,
        params={"orientation_sigma": 0.35, "gyro_sigma": 1.0, "accel_sigma": 2.0},
    )
    imitation_gyro = RewardTermCfg(func=_term(_TASK_MODULE, "imitation_gyro_reward"), weight=0.5, params={"sigma": 1.0})
    imitation_joint = RewardTermCfg(
        func=_term(_TASK_MODULE, "imitation_joint_reward"), weight=0.5, params={"sigma": 0.35}
    )
    imitation_velocity = RewardTermCfg(
        func=_term(_TASK_MODULE, "imitation_velocity_reward"), weight=0.5, params={"sigma": 0.5}
    )
    imitation_rgb = RewardTermCfg(func=_term(_TASK_MODULE, "imitation_rgb_reward"), weight=0.25, params={"sigma": 0.25})
    imitation_event = RewardTermCfg(
        func=_term(_TASK_MODULE, "imitation_event_reward"), weight=0.25, params={"sigma": 1.0}
    )


@configclass
class MyExoskeletonImitationEnvCfg(MyExoskeletonFlatEnvCfg):
    """模仿任务总装层，用扩展配置替换父类对应的 scene/obs/command/reward。"""

    scene: MyExoskeletonImitationSceneCfg = MyExoskeletonImitationSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: MyExoskeletonImitationObservationsCfg = MyExoskeletonImitationObservationsCfg()
    commands: MyExoskeletonImitationCommandsCfg = MyExoskeletonImitationCommandsCfg()
    rewards: MyExoskeletonImitationRewardsCfg = MyExoskeletonImitationRewardsCfg()

    # 这些字段可由命令行转成 Hydra 覆盖，例如 --task_id BT02 最终会设置
    # env.imitation_task_id=BT02，而无需改源码。
    imitation_dataset_path: str | None = None
    # 记录数据中 IMU 和关节通道的方向/符号标定。未设置时，加载器使用仓库级别的
    # `sensor_coordinate_mapping.yaml` 映射文件。
    imitation_calibration_path: str | None = None
    imitation_task_id: str = "all"
    imitation_random_task_sampling: bool = True
    # 可选传感器开关。默认保持启用，使已有训练命令继续遵守多模态契约。场景声明仍保留供
    # 配置检查；运行时会在构造 InteractiveScene 前移除已禁用实体。观测项仍返回固定
    # shape 的零张量并把通道标为无效，因此传感器消融时 manager/rl_games 空间保持稳定。
    use_imu: bool = True
    use_rgb: bool = True
    use_event: bool = True
    use_contact: bool = True
    # 每段目标保留的连续 50 Hz 帧数。默认 32 帧约等于 0.64 s；若设为 1，教师目标
    # 没有时间变化，动态动作模仿就退化为随机静态姿势匹配。
    imitation_window_size: int = 32
    imitation_rgb_features: bool = True
    imitation_event_features: bool = True
    # 周期步态可在序列末尾循环；起停/场景任务则在最后一帧保持到 episode reset。
    # 否则非周期记录会突然跳回第一帧，策略可能把这种不真实跳变也学进去。
    imitation_sequence_loop: bool = True
    imitation_non_cyclic_tasks: tuple[str, ...] = (
        "BT01",
        "BT05",
        "AT01",
        "AT02",
        "AT03",
        "AT04",
        "AT05",
    )
    # 提供的 DVXplorer 记录分辨率为 320x240；元组顺序是 (height, width)。
    imitation_event_source_size: tuple[int, int] = (240, 320)
    store_raw_camera_frames: bool = False
    imu_mapping_output_path: str | None = None
    # 可选的回放期遥测。紧凑特征统计适用于 4096 个环境；启用
    # ``store_raw_camera_frames`` 后，原始 RGB 帧仍会保存在 env 缓存中，供单环境
    # Kit/Viser 检查。
    diagnostics_enabled: bool = False
    diagnostics_interval: int = 20
    diagnostics_output_path: str | None = None

    def __post_init__(self):
        """先应用父类 50 Hz/导出检查，再设置多模态空间和数据集选项。"""

        super().__post_init__()
        # ObservationManager 会根据上面的命名组创建实际空间；这里的整数同时向配置查看
        # 工具明确声明总维度，避免仍显示父类的 49 维。动作空间继承后仍为 8。
        self.observation_space = POLICY_OBS_DIM + IMU_COUNT * IMU_FEATURES + RGB_FEATURES + EVENT_FEATURES
        self.state_space = CRITIC_OBS_DIM
        self.action_space = len(ACTUATED_JOINT_NAMES)
        self.scene.env_spacing = max(float(self.scene.env_spacing), 4.0)
        if self.commands.base_velocity.task_id == "all":
            self.commands.base_velocity.task_id = self.imitation_task_id
        self.commands.base_velocity.random_task_sampling = self.imitation_random_task_sampling
        self.commands.base_velocity.window_size = self.imitation_window_size
        # ``imitation_*_features`` 是旧版数据加载器开关。保留它们作为额外的关闭方式，
        # 同时让公共传感器标志决定命令行 ``env.use_*`` 覆盖的最终结果。
        self.imitation_rgb_features = bool(self.imitation_rgb_features and self.use_rgb)
        self.imitation_event_features = bool(self.imitation_event_features and self.use_event)
        if self.diagnostics_interval <= 0:
            raise ValueError("diagnostics_interval must be positive")


# 外部脚本使用规格要求的名称；同时保留一个简短别名。
MyExoskeletonImitationEnvConfig = MyExoskeletonImitationEnvCfg

__all__ = [
    "MyExoskeletonImitationCommandsCfg",
    "MyExoskeletonImitationEnvCfg",
    "MyExoskeletonImitationEnvConfig",
    "MyExoskeletonImitationRewardsCfg",
    "MyExoskeletonImitationSceneCfg",
]
