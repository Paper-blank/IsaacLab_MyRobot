"""Isaac Lab 3.0 configuration for vision/IMU motion imitation.

The locomotion terms from :mod:`my_exoskeleton_env_cfg` remain active.  This
configuration adds six mapped IMUs, two fixed RGB cameras, two fixed DVS
camera prims, and compact multimodal observation groups that can be consumed
by the stock rl_games PPO MLP at 4096 environments.
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
    return f"{module}:{name}"


_INSPECTABLE_IMU_CLASS = "isaaclab_assets.sensors.inspectable_imu:InspectableImu"


def _event_cfg(name: str, side: str):
    """Use EVIS when installed, otherwise keep an explicit invalid adapter."""

    if event_camera_plugin_available():
        return make_event_camera_cfg(name, side)  # type: ignore[arg-type]
    return make_deferred_event_camera_cfg(name, side)  # type: ignore[arg-type]


@configclass
class MyExoskeletonImitationSceneCfg(MyExoskeletonSceneCfg):
    """Flat 2.4 m channel with fixed stereo RGB/DVS rigs and mapped IMUs."""

    # ``TerrainImporterCfg(terrain_type="plane")`` is intentionally kept as
    # the physical ground.  It is infinite and has no visible channel
    # boundary, though, so Viser/Newton can only show a blank plane.  These
    # lightweight, visual-only cuboids provide an explicit local channel
    # footprint for both the interactive viewer and the RGB cameras.  They
    # are cloned with each environment and do not add contacts or dynamics.
    channel_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Channel/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(CHANNEL_LENGTH_M, 2.0 * DEFAULT_CHANNEL_HALF_WIDTH_M, 0.012),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.16, 0.20, 0.24), roughness=0.92
            ),
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
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CAMERA_X_M, DEFAULT_CHANNEL_HALF_WIDTH_M + 0.025, 0.0225)
        ),
    )
    channel_right_edge = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Channel/RightEdge",
        spawn=sim_utils.CuboidCfg(
            size=(CHANNEL_LENGTH_M, 0.045, 0.045),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.66, 0.12), roughness=0.75),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CAMERA_X_M, -(DEFAULT_CHANNEL_HALF_WIDTH_M + 0.025), 0.0225)
        ),
    )

    # Replace the six legacy names with the canonical recorder IDs.  The
    # original flat task keeps its old aliases for backwards-compatible play.
    imu_rthigh = None
    imu_rleg = None
    imu_rfoot = None
    imu_lthigh = None
    imu_lleg = None
    imu_lfoot = None

    # Recorder order is interleaved by side and joint level: 01/02 are the
    # left/right ankles, 03/04 the left/right knees, and 05/06 the left/right
    # hips.  Keeping the scene names identical to recorder IDs makes
    # target_imu[:, sensor_index] unambiguous.
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
    # EVIS uses these exact names to find its asynchronous stream.  The
    # fallback adapter keeps the names/poses but marks its stream invalid.
    left_event_camera = _event_cfg("left_event_camera", "left")
    right_event_camera = _event_cfg("right_event_camera", "right")

    # Metadata properties are deliberately not config fields.  The scene
    # collector treats every extra field as an entity and would otherwise try
    # to expand these floats as USD prim paths.
    @property
    def channel_length_m(self) -> float:
        return CHANNEL_LENGTH_M

    @property
    def channel_half_width_m(self) -> float:
        return DEFAULT_CHANNEL_HALF_WIDTH_M


@configclass
class MyExoskeletonImitationCommandsCfg(MyExoskeletonCommandsCfg):
    """Velocity command fields used to select and randomize recording tasks."""

    @configclass
    class BaseVelocityCfg(MyExoskeletonCommandsCfg.BaseVelocityCfg):
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
    """Separate top-level groups preserve ``obs["imu"]``/camera contracts."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=_term(_FLAT_TASK_MODULE, "exoskeleton_observation"))

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class ImuCfg(ObservationGroupCfg):
        imu = ObservationTermCfg(func=_term(_TASK_MODULE, "imu_observation"))

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CameraRgbCfg(ObservationGroupCfg):
        camera_rgb = ObservationTermCfg(func=_term(_TASK_MODULE, "camera_rgb_observation"))

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class EventCameraCfg(ObservationGroupCfg):
        event_camera = ObservationTermCfg(func=_term(_TASK_MODULE, "event_camera_observation"))

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=_term(_FLAT_TASK_MODULE, "exoskeleton_critic_observation"))

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    imu: ImuCfg = ImuCfg()
    camera_rgb: CameraRgbCfg = CameraRgbCfg()
    event_camera: EventCameraCfg = EventCameraCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class MyExoskeletonImitationRewardsCfg(MyExoskeletonRewardsCfg):
    """Positive multimodal similarity terms layered on the gait rewards."""

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
    """Complete Vision + IMU + Motion Imitation environment configuration."""

    scene: MyExoskeletonImitationSceneCfg = MyExoskeletonImitationSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: MyExoskeletonImitationObservationsCfg = MyExoskeletonImitationObservationsCfg()
    commands: MyExoskeletonImitationCommandsCfg = MyExoskeletonImitationCommandsCfg()
    rewards: MyExoskeletonImitationRewardsCfg = MyExoskeletonImitationRewardsCfg()

    # Hydra can override these values without changing the command generator.
    imitation_dataset_path: str | None = None
    # Direction/sign calibration for recorded IMU and joint channels.  When
    # unset, the loader uses the repository-level sensor_coordinate_mapping.yaml.
    imitation_calibration_path: str | None = None
    imitation_task_id: str = "all"
    imitation_random_task_sampling: bool = True
    # Optional sensor switches. Keep these enabled by default so existing
    # training commands retain the multimodal contract. The scene declarations
    # remain available for config inspection; the runtime removes disabled
    # entities immediately before InteractiveScene construction. Observation
    # terms still return fixed-shape zero tensors and mark the channel invalid,
    # so the manager/rl_games spaces stay stable across sensor ablations.
    use_imu: bool = True
    use_rgb: bool = True
    use_event: bool = True
    use_contact: bool = True
    # Number of 50 Hz target frames kept per episode segment.  A value of one
    # would reduce dynamic imitation to a static pose reward.
    imitation_window_size: int = 32
    imitation_rgb_features: bool = True
    imitation_event_features: bool = True
    # Periodic gait recordings can wrap at the end of a window.  Transition
    # and scenario recordings (BT05/AT01-AT05) should instead hold their last
    # target frame until the episode resets, otherwise every episode ends by
    # jumping back to the start/stop action and the policy learns that loop.
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
    # Supplied DVXplorer recordings are 320x240; tuple order is (height, width).
    imitation_event_source_size: tuple[int, int] = (240, 320)
    store_raw_camera_frames: bool = False
    imu_mapping_output_path: str | None = None
    # Optional play-time telemetry.  The compact feature statistics are safe
    # for 4096 environments; raw RGB frames remain available in the env cache
    # for one-environment Kit/Viser inspection when ``store_raw_camera_frames``
    # is enabled.
    diagnostics_enabled: bool = False
    diagnostics_interval: int = 20
    diagnostics_output_path: str | None = None

    def __post_init__(self):
        super().__post_init__()
        # The manager creates the actual Gym spaces from the named groups.  The
        # scalar declarations still document the policy/critic contract and
        # keep config-only tools from reporting the legacy 49-D size.
        self.observation_space = POLICY_OBS_DIM + IMU_COUNT * IMU_FEATURES + RGB_FEATURES + EVENT_FEATURES
        self.state_space = CRITIC_OBS_DIM
        self.action_space = len(ACTUATED_JOINT_NAMES)
        self.scene.env_spacing = max(float(self.scene.env_spacing), 4.0)
        if self.commands.base_velocity.task_id == "all":
            self.commands.base_velocity.task_id = self.imitation_task_id
        self.commands.base_velocity.random_task_sampling = self.imitation_random_task_sampling
        self.commands.base_velocity.window_size = self.imitation_window_size
        # ``imitation_*_features`` are the legacy data-loader switches.  Keep
        # them as an additional opt-out while making the public sensor flags
        # authoritative for the command-line ``env.use_*`` overrides.
        self.imitation_rgb_features = bool(self.imitation_rgb_features and self.use_rgb)
        self.imitation_event_features = bool(self.imitation_event_features and self.use_event)
        if self.diagnostics_interval <= 0:
            raise ValueError("diagnostics_interval must be positive")


# The requested filename is used by external scripts; keep a short alias too.
MyExoskeletonImitationEnvConfig = MyExoskeletonImitationEnvCfg

__all__ = [
    "MyExoskeletonImitationCommandsCfg",
    "MyExoskeletonImitationEnvCfg",
    "MyExoskeletonImitationEnvConfig",
    "MyExoskeletonImitationRewardsCfg",
    "MyExoskeletonImitationSceneCfg",
]
