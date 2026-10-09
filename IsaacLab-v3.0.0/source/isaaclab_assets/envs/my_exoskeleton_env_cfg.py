"""Configuration for the my_exoskeleton velocity-tracking task.

The URDF contains eight actuated revolute joints, six IMUs (on the thigh, shank,
and foot links), and one contact sensor on each foot.  Isaac Lab creates the
runtime sensors from the corresponding USD body paths after URDF conversion.
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

# Sensor implementation modules backed by Omniverse may only be imported after
# Kit has started.  Keep this as a resolvable string so configuration discovery,
# ``--help``, and non-Kit tests remain importable.
_INSPECTABLE_IMU_CLASS = "isaaclab_assets.sensors.inspectable_imu:InspectableImu"
_NESTED_CONTACT_SENSOR_CLASS = "isaaclab_assets.sensors.contact_sensor:NestedBodyContactSensor"
_TASK_MODULE = "isaaclab_assets.tasks.my_exoskeleton_flat"


def _task_term(name: str) -> str:
    """Return a Kit-safe deferred callable reference for a task term."""

    return f"{_TASK_MODULE}:{name}"


URDF_PATH = str(Path(__file__).resolve().parents[1] / "data/my_robot/urdf/my_exoskeleton.urdf")
# First-stage walking target in the robot base frame [m/s].
FIXED_FORWARD_SPEED_M_S = 0.2
# Curriculum boundaries are expressed in global control steps by Isaac Lab.
# The PPO rollout contains 32 control steps, so 500 PPO iterations correspond
# to 16,000 control steps.  Each subsequent gait phase receives another 500
# iterations under the default recurrent PPO configuration.
CURRICULUM_ITERATIONS_PER_PHASE = 500
PPO_HORIZON_LENGTH = 32
PHASE_2_START_STEP = CURRICULUM_ITERATIONS_PER_PHASE * PPO_HORIZON_LENGTH
PHASE_3_START_STEP = 2 * CURRICULUM_ITERATIONS_PER_PHASE * PPO_HORIZON_LENGTH
MAX_CURRICULUM_SPEED_M_S = 4.0


@configclass
class MyExoskeletonSceneCfg(InteractiveSceneCfg):
    """Scene entities, including the URDF articulation and runtime sensors."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=URDF_PATH,
            fix_base=False,
            merge_fixed_joints=True,
            make_instanceable=False,
            activate_contact_sensors=True,
            # Keep self-collision enabled as required by the task.  The
            # imported STL convex hulls overlap at the nominal joint pose, so
            # dedicated low-poly collision meshes and adjacent-link filtering
            # are still required before treating the visual result as final.
            self_collision=True,
            # Rebuild the cached USD when the URDF importer settings change.
            # Without this flag an older payload can retain a disabled
            # self-collision attribute even though the runtime articulation
            # properties below request self-collisions.
            force_usd_conversion=True,
            # Full CAD triangle meshes are too concave/noisy for stable
            # self-contact.  Convex decomposition keeps self-collision on
            # while avoiding the large interpenetration impulses that show up
            # as apparent link deformation in the renderer.
            collision_type="Convex Decomposition",
            merge_mesh=False,
            # Isaac Lab 3.0 moved PhysX-only schema cfgs to
            # ``isaaclab_physx.sim.schemas``.  Use the concrete class directly
            # so self-collision and solver fields are authored in the
            # ``physxArticulation`` namespace without a forwarding alias.
            articulation_props=PhysxArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=2
            ),
        ),
        # The URDF limits below already include a small anatomical extension
        # margin.  Keep the Isaac Lab soft region equal to those asymmetric
        # limits; shrinking around the midpoint would move the zero/extension
        # pose outside the soft interval on a mirrored knee.
        soft_joint_pos_limit_factor=1.0,
        # The URDF foot meshes are already authored with their sole near z=0 in
        # the root frame.  Keep the root just above the plane to avoid spawning
        # the robot 1.25 m in the air and dropping on every reset.
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)),
        actuators={
            "exoskeleton_pd": ImplicitActuatorCfg(
                joint_names_expr=list(ACTUATED_JOINT_NAMES),
                stiffness=80.0,
                damping=4.0,
                joint_effort_limit={name: 60.0 for name in ACTUATED_JOINT_NAMES},
                joint_velocity_limit={name: 3.14 for name in ACTUATED_JOINT_NAMES},
            )
        },
    )

    # The URDF IMUs are on these links.  Root state is used as a physically
    # equivalent base estimate because the URDF has no pelvis/torso IMU.
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
        # Net contact force is sufficient for the policy.  PhysX requires an
        # explicit filter set when contact-point coordinates are requested.
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
    """Gait-aware velocity command generator settings."""

    @configclass
    class BaseVelocityCfg(mdp.UniformVelocityCommandCfg):
        """Velocity command plus optional on-device sensor reference playback."""

        class_type = _task_term("GaitVelocityCommand")
        reference_data_path: str | None = None
        reference_target_dt_s: float = CONTROL_DT_S
        reference_loop: bool = True
        reference_random_start: bool = False

    base_velocity = BaseVelocityCfg(
        class_type=_task_term("GaitVelocityCommand"),
        asset_name="robot",
        # Targets persist long enough for the 1.5 s W2S and 2.5 s R2W tests.
        # The gait classifier itself is updated at 10 Hz by the command class.
        resampling_time_range=(3.0, 6.0),
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=False,
        # The upstream defaults (0.5 m/s, 0.4 rad/s) would classify a robot
        # standing still under the phase-one 0.2 m/s command as successful.
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
    """Gait-conditioned actor observations and privileged critic state."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Concatenated policy observation."""

        state = ObservationTermCfg(func=_task_term("exoskeleton_observation"))

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Actor state augmented with simulator-only contact and dynamics data."""

        state = ObservationTermCfg(func=_task_term("exoskeleton_critic_observation"))

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    critic: CriticCfg = CriticCfg()


@configclass
class MyExoskeletonActionsCfg:
    """Action manager terms for incremental joint-position control."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ACTUATED_JOINT_NAMES),
        scale=dict(JOINT_ACTION_SCALE_RAD),
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class MyExoskeletonRewardsCfg:
    """Shared and mutually-exclusive gait rewards adapted from the paper."""

    # The terminal penalty is retained as a practical proxy for the source
    # paper's collision penalty because this model only senses its two feet.
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
    alive = RewardTermCfg(func=mdp.is_alive, weight=0.15)
    track_linear_velocity = RewardTermCfg(
        func=_task_term("tracking_linear_velocity_reward"),
        weight=1.5,
        params={"std": 0.5, "activity_gate_speed": 0.1, "activity_gate_command_max": 0.25},
    )
    # The paper's broad exponential tracker alone gives a stationary robot
    # exp(-0.2^2 / 0.5^2) = 0.852 in phase one.  This auxiliary window is
    # zero at rest and restores the exploration pressure needed to learn a
    # first walking cycle before later command ranges are introduced.
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
    # Disabled in practice when no reference path is configured: the task term
    # returns an all-zero tensor.  Set ``reference_data_path`` and tune this
    # weight for a recording-specific retraining run.
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
            # Right-leg then left-leg body ordering is required by the
            # morphology-adapted side-momentum term.
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

    # Walking rewards.
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

    # Standing rewards.
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

    # Walk-to-stand rewards.
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

    # Running rewards.
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

    # Run-to-walk rewards.
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
    """Episode reset and paper-style dynamics randomization."""

    randomize_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
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
        interval_range_s=(3.0, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (0.0, 0.0), "y": (-0.3, 0.3), "yaw": (-0.2, 0.2)},
        },
    )


@configclass
class MyExoskeletonCurriculumCfg:
    """Three-stage gait curriculum based on global control steps."""

    gait_commands = CurriculumTermCfg(
        func=_task_term("advance_exoskeleton_curriculum"),
        params={
            "phase2_start": PHASE_2_START_STEP,
            "phase3_start": PHASE_3_START_STEP,
            # Set this to prior_iterations * horizon_length when resuming;
            # RL-Games checkpoints do not contain the environment step count.
            "step_offset": 0,
            "walk_speed": FIXED_FORWARD_SPEED_M_S,
            "transition_max_speed": 0.8,
            "multi_gait_max_speed": MAX_CURRICULUM_SPEED_M_S,
        },
    )


@configclass
class MyExoskeletonTerminationsCfg:
    """Episode termination conditions."""

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
    """Complete gait-conditioned flat-ground exoskeleton configuration."""

    scene: MyExoskeletonSceneCfg = MyExoskeletonSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: MyExoskeletonObservationsCfg = MyExoskeletonObservationsCfg()
    actions: MyExoskeletonActionsCfg = MyExoskeletonActionsCfg()
    commands: MyExoskeletonCommandsCfg = MyExoskeletonCommandsCfg()
    rewards: MyExoskeletonRewardsCfg = MyExoskeletonRewardsCfg()
    terminations: MyExoskeletonTerminationsCfg = MyExoskeletonTerminationsCfg()
    # Managers require config objects even when no terms are enabled.
    events: MyExoskeletonEventsCfg = MyExoskeletonEventsCfg()
    curriculum: MyExoskeletonCurriculumCfg = MyExoskeletonCurriculumCfg()
    episode_length_s = 20.0
    # Optional sensor switches are shared with the imitation child config so
    # the convenience CLI remains valid for the legacy flat task as well.
    # The flat scene has no RGB/Event entities; those flags are accepted as
    # no-op compatibility controls, while IMU/contact terms honor them.
    use_imu: bool = True
    use_rgb: bool = True
    use_event: bool = True
    use_contact: bool = True
    # Optional structured playback output.  The actor/action contract remains
    # eight-dimensional; the recorder exposes a 24-dimensional trajectory
    # result (angle, velocity, angular acceleration) for the selected env.
    ruleresult_enabled: bool = False
    ruleresult_interval: int = 1
    ruleresult_env_id: int = 0
    ruleresult_output_path: str | None = None
    ruleresult_output_format: str = "jsonl"
    ruleresult_protocol_config_path: str | None = None
    # ``ruleresult_dt_s`` is the nominal control period/fallback.  The
    # recorder can instead use simulator time or a monotonic wall clock for
    # each finite difference when deployment timing is not fixed.
    ruleresult_dt_s: float = CONTROL_DT_S
    ruleresult_time_mode: str = "simulation"
    ruleresult_min_dt_s: float = 1.0e-6
    ruleresult_max_dt_s: float = 10.0
    ruleresult_prediction_horizon: int = 0
    ruleresult_prediction_mode: str = "none"
    ruleresult_prediction_dt_s: float | None = None

    def __post_init__(self):
        """Set simulation timing and the declared Gym dimensions."""
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
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


# Backwards-compatible name requested by the task specification.
MyExoskeletonEnvCfg = MyExoskeletonFlatEnvCfg
