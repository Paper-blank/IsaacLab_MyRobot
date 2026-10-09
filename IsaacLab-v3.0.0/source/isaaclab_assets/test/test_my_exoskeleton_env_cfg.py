# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""外骨骼基础环境配置、奖励和安全边界的回归测试。

本文件保护环境真正交给训练器的契约：初始姿态与 reset、8关节动作顺序和缩放、课程
切换步数、奖励权重、左右膝镜像限位、接触模式、前进奖励和足部朝向。多数测试使用
最小假环境直接调用奖励函数，目的是在不启动 Isaac Sim 的情况下验证数学和路由逻辑；
它们不是在声称假张量能够代替最终的整机仿真测试。
"""

import torch
from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

from isaaclab.managers import SceneEntityCfg

from isaaclab_assets.envs.my_exoskeleton_env_cfg import (
    CURRICULUM_ITERATIONS_PER_PHASE,
    FIXED_FORWARD_SPEED_M_S,
    PHASE_2_START_STEP,
    PHASE_3_START_STEP,
    PPO_HORIZON_LENGTH,
    MyExoskeletonFlatEnvCfg,
)
from isaaclab_assets.tasks import my_exoskeleton_flat
from isaaclab_assets.tasks.robot_contract import (
    ACTUATED_JOINT_NAMES,
    JOINT_ACTION_SCALE_RAD,
    KNEE_ARTICULATION_LIMITS_RAD,
    KNEE_ARTICULATION_SIGNS,
    KNEE_JOINT_NAMES,
)


def test_exoskeleton_reset_restores_scene_and_starts_near_ground():
    """倒地自动重置应保持启用，默认根节点高度也应与脚底平面对齐。"""
    cfg = MyExoskeletonFlatEnvCfg()

    assert cfg.scene.robot.init_state.pos == (0.0, 0.0, 0.01)
    assert cfg.events.reset_scene_to_default.mode == "reset"
    assert cfg.events.reset_scene_to_default.params == {"reset_joint_targets": True}
    assert cfg.events.reset_scene_to_default.func.__name__ == "reset_scene_to_default"
    assert cfg.terminations.fallen.func.__name__ == "bad_orientation"
    assert cfg.terminations.fallen.time_out is False
    assert cfg.scene.robot.spawn.self_collision is True
    assert cfg.scene.robot.spawn.force_usd_conversion is True
    assert cfg.scene.robot.spawn.collision_type == "Convex Decomposition"
    assert cfg.scene.robot.spawn.merge_mesh is False
    # Isaac Lab 3.0 已把 PhysX 专用 schema 配置移出 ``isaaclab.sim``。这里直接检查
    # 目标类，防止弃用的转发别名悄悄重新引入后端不完整的自碰撞配置。
    assert isinstance(cfg.scene.robot.spawn.articulation_props, PhysxArticulationRootPropertiesCfg)
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is True
    assert cfg.scene.left_foot_contact.class_type == ("isaaclab_assets.sensors.contact_sensor:NestedBodyContactSensor")
    assert cfg.scene.right_foot_contact.class_type == ("isaaclab_assets.sensors.contact_sensor:NestedBodyContactSensor")


def test_action_contract_preserves_canonical_joint_order_and_per_joint_scales():
    """策略动作顺序必须与硬件和参考数据共同约定的关节顺序一致。"""
    cfg = MyExoskeletonFlatEnvCfg()
    action = cfg.actions.joint_pos
    assert action.joint_names == list(ACTUATED_JOINT_NAMES)
    assert action.preserve_order is True
    assert action.scale == JOINT_ACTION_SCALE_RAD


def test_ruleresult_output_is_opt_in_and_keeps_the_eight_dimensional_actor_contract():
    """回放结果可以是 24 维，但不能因此改变 PPO 的 8 维动作空间。"""

    cfg = MyExoskeletonFlatEnvCfg()

    assert cfg.action_space == len(ACTUATED_JOINT_NAMES)
    assert cfg.ruleresult_enabled is False
    assert cfg.ruleresult_interval == 1
    assert cfg.ruleresult_env_id == 0
    assert cfg.ruleresult_dt_s == 0.02
    assert cfg.ruleresult_time_mode == "simulation"
    assert cfg.ruleresult_prediction_horizon == 0
    assert cfg.ruleresult_prediction_mode == "none"


def test_curriculum_switches_after_500_ppo_iterations():
    """把 PPO 迭代数换算为环境步数，锁定每 500 轮才切换一次课程的边界。"""
    assert CURRICULUM_ITERATIONS_PER_PHASE == 500
    assert PPO_HORIZON_LENGTH == 32
    assert PHASE_2_START_STEP == 500 * 32
    assert PHASE_3_START_STEP == 1000 * 32


def test_exoskeleton_uses_shared_and_routed_paper_reward_weights():
    """安全惩罚以及代表性的共享/分步态奖励权重都应明确配置。"""
    cfg = MyExoskeletonFlatEnvCfg()

    assert cfg.rewards.termination_penalty.func.__name__ == "is_terminated"
    assert cfg.rewards.termination_penalty.weight == -200.0
    assert cfg.rewards.track_linear_velocity.weight == 1.5
    assert cfg.rewards.fixed_forward_walk.weight == 3.0
    assert cfg.rewards.action_rate.weight == -0.01
    assert cfg.rewards.hip_position_deviation.weight == -1.0
    assert cfg.rewards.leg_momentum_balance.weight == 5.0
    assert cfg.rewards.leg_mechanical_power.weight == -0.3
    assert cfg.rewards.leg_phase_symmetry.weight == 2.0
    assert cfg.rewards.leg_amplitude_match.weight == 1.0
    assert cfg.rewards.sensor_reference_imitation.weight == 1.0
    assert cfg.commands.base_velocity.reference_random_start is False
    assert cfg.rewards.walk_contact.weight == 1.0
    assert cfg.rewards.w2s_contact.params["allow_double_support"] is True
    assert cfg.rewards.w2s_contact.params["double_support_ramp_s"] == 1.5
    assert cfg.rewards.stand_contact.weight == 2.5
    assert cfg.rewards.stand_foot_drag.weight == -0.2
    assert cfg.rewards.run_contact.params["flight_window"] == 0.35
    assert cfg.rewards.run_swing_height.weight == -25.0


def test_knee_limits_are_mirrored_and_use_side_specific_terms():
    """全局 ``[-2.25, 2.25]`` 边界不得让膝关节向反方向弯折。"""
    cfg = MyExoskeletonFlatEnvCfg()

    assert KNEE_ARTICULATION_SIGNS == {"Rthigh_to_Rleg": -1.0, "Lthigh_to_Lleg": 1.0}
    assert KNEE_ARTICULATION_LIMITS_RAD == {
        "Rthigh_to_Rleg": (-2.25, 0.05),
        "Lthigh_to_Lleg": (-0.05, 2.25),
    }
    assert cfg.scene.robot.soft_joint_pos_limit_factor == 1.0
    assert cfg.rewards.knee_human_range.params["bounds_by_joint"] == KNEE_ARTICULATION_LIMITS_RAD
    assert cfg.terminations.knee_out_of_human_range.func == (
        "isaaclab_assets.tasks.my_exoskeleton_flat:knee_out_of_human_range"
    )
    assert cfg.terminations.knee_out_of_human_range.params["bounds_by_joint"] == KNEE_ARTICULATION_LIMITS_RAD
    assert cfg.scene.robot.actuators["exoskeleton_pd"].joint_velocity_limit[KNEE_JOINT_NAMES[0]] == 3.14
    assert cfg.scene.robot.actuators["exoskeleton_pd"].joint_effort_limit[KNEE_JOINT_NAMES[1]] == 60.0


def test_leg_momentum_reward_uses_explicit_right_left_body_pairing():
    """动量项不得简单地把完整 PhysX 刚体数组从中间一分为二。"""
    cfg = MyExoskeletonFlatEnvCfg()
    asset_cfg = cfg.rewards.leg_momentum_balance.params["asset_cfg"]

    assert asset_cfg.body_names == [
        "R_turn",
        "Rthigh",
        "Rleg",
        "Rfoot",
        "Lturn",
        "Lthigh",
        "Lleg",
        "Lfoot",
    ]
    assert asset_cfg.preserve_order is True


def test_leg_momentum_reward_is_negative_centroidal_residual():
    """论文公式 (2) 必须是成本项，不能把双腿晃动误当成活动奖励。"""

    class FakeData:
        body_mass = torch.ones(1, 2)
        body_com_pos_w = torch.tensor([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        body_com_vel_w = torch.tensor([[0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]).unsqueeze(0)
        body_inertia = torch.zeros(1, 2, 9)
        body_link_quat_w = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
        joint_vel = torch.zeros(1, 8)
        root_lin_vel_b = torch.zeros(1, 3)
        root_ang_vel_b = torch.zeros(1, 3)
        root_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    class FakeRobot:
        data = FakeData()

    class FakeEnv:
        scene = {"robot": FakeRobot()}

    resolved_cfg = SceneEntityCfg("robot", body_ids=[0, 1])
    reward = my_exoskeleton_flat.leg_momentum_balance_reward(FakeEnv(), asset_cfg=resolved_cfg, momentum_scale=2.0)

    assert torch.isclose(reward, torch.tensor([-1.4])).all()


def test_centroidal_momentum_rotates_physx_link_inertia_to_world_frame():
    """用世界系角速度计算动量时，必须先把连杆惯量旋转到世界坐标系。"""

    angle = torch.tensor(torch.pi / 4.0)

    class FakeData:
        body_mass = torch.ones(1, 2)
        body_com_pos_w = torch.zeros(1, 2, 3)
        body_com_vel_w = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        # PhysX 按列优先展开；这个对角矩阵在该展开规则下数值顺序不变。
        body_inertia = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], [0.0] * 9]])
        body_link_quat_w = torch.tensor(
            [[[0.0, torch.sin(angle / 2.0), 0.0, torch.cos(angle / 2.0)], [0.0, 0.0, 0.0, 1.0]]]
        )

    total_z, _ = my_exoskeleton_flat._centroidal_angular_momentum_z(FakeData())

    # ``R_y(pi/4) diag(1,2,3) R_y(pi/4)^T`` 的 ``I_zx=1``，所以最终结果为
    # ``L_z=I_zx*omega_x=1``，正好可用于核对坐标变换是否正确。
    assert torch.isclose(total_z, torch.tensor([1.0]), atol=1.0e-5).all()


def test_first_curriculum_phase_starts_with_fixed_forward_walk_command():
    """第一阶段应先保留稳定的固定前进任务，再逐步加入步态过渡。"""
    cfg = MyExoskeletonFlatEnvCfg()
    command = cfg.commands.base_velocity

    assert command.rel_standing_envs == 0.0
    assert command.vel_xy_success_threshold == 0.05
    assert command.vel_yaw_success_threshold == 0.10
    assert command.ranges.lin_vel_x == (FIXED_FORWARD_SPEED_M_S, FIXED_FORWARD_SPEED_M_S)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert command.resampling_time_range == (3.0, 6.0)
    assert str(command.class_type) == "isaaclab_assets.tasks.my_exoskeleton_flat:GaitVelocityCommand"
    assert cfg.curriculum.gait_commands.params["walk_speed"] == FIXED_FORWARD_SPEED_M_S
    assert cfg.curriculum.gait_commands.params["step_offset"] == 0


def test_fixed_forward_walk_reward_requires_forward_motion_at_target_speed():
    """给定七种代表性运动，只有按目标速度直行的环境才能得到满分。

    这个黄金向量专门防止“保持站立也有高奖励”的捷径，同时确认倒退、过快、横移和
    原地转向都不会被当作完成固定正前方行走任务。
    """
    target_speed = 0.2

    class FakeData:
        # 七行依次表示：目标速度直行、站立、倒退、过快、速度不足、横移、转向。
        root_lin_vel_b = torch.tensor(
            [
                [target_speed, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-0.1, 0.0, 0.0],
                [2.0 * target_speed, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [target_speed, 0.15, 0.0],
                [target_speed, 0.0, 0.5],
            ]
        )
        root_ang_vel_b = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5],
            ]
        )
        root_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(7, 1)

    class FakeRobot:
        data = FakeData()

    class FakeEnv:
        scene = {"robot": FakeRobot()}

    reward = my_exoskeleton_flat.fixed_forward_walk_reward(
        FakeEnv(),
        target_speed=target_speed,
        speed_tolerance=0.2,
        lateral_speed_std=0.15,
        yaw_rate_std=0.5,
    )

    assert reward.shape == (7,)
    assert torch.isclose(reward[0], torch.tensor(1.0))
    assert torch.equal(reward[1:4], torch.zeros(3))
    assert torch.isclose(reward[4], torch.tensor(0.75))
    assert torch.isclose(reward[5], torch.exp(torch.tensor(-1.0)))
    assert torch.isclose(reward[6], torch.exp(torch.tensor(-1.0)))


def test_low_speed_velocity_tracking_gate_removes_stationary_shortcut():
    """第一阶段的跟踪奖励不得给零前进速度满分。"""

    class FakeData:
        root_lin_vel_b = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        root_ang_vel_b = torch.zeros(2, 3)
        root_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)

    class FakeRobot:
        data = FakeData()

    class FakeCommandManager:
        @staticmethod
        def get_command(_name):
            return torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])

    class FakeEnv:
        scene = {"robot": FakeRobot()}
        command_manager = FakeCommandManager()

    reward = my_exoskeleton_flat.tracking_linear_velocity_reward(
        FakeEnv(), std=0.5, activity_gate_speed=0.1, activity_gate_command_max=0.25
    )

    assert torch.isclose(reward[0], torch.tensor(0.0))
    assert torch.isclose(reward[1], torch.tensor(1.0))


def test_phase_contact_requires_the_complete_single_support_pattern(monkeypatch):
    """给定双脚/单脚接触组合，普通行走只奖励与相位一致的完整单支撑模式。"""
    gait_modes = torch.full((3,), 1, dtype=torch.long)

    class FakeCommandTerm:
        gait_mode = gait_modes
        gait_phase = torch.tensor([0.5, 0.5, -0.5])

    class FakeCommandManager:
        @staticmethod
        def get_term(_name):
            return FakeCommandTerm()

    class FakeEnv:
        command_manager = FakeCommandManager()

    monkeypatch.setattr(
        my_exoskeleton_flat,
        "_foot_contacts",
        lambda _env: (torch.tensor([10.0, 10.0, 0.0]), torch.tensor([10.0, 0.0, 10.0])),
    )

    reward = my_exoskeleton_flat.phase_contact_reward(FakeEnv(), gait=1)

    assert torch.equal(reward, torch.tensor([0.0, 1.0, 1.0]))


def test_phase_contact_allows_run_flight_but_rejects_double_support(monkeypatch):
    """跑步允许与相位一致的腾空，同时仍应保留单脚支撑判定。"""
    gait_modes = torch.full((4,), 3, dtype=torch.long)

    class FakeCommandTerm:
        gait_mode = gait_modes
        gait_phase = torch.tensor([0.0, torch.pi / 2.0, -torch.pi / 2.0, 0.0])

    class FakeCommandManager:
        @staticmethod
        def get_term(_name):
            return FakeCommandTerm()

    class FakeEnv:
        command_manager = FakeCommandManager()

    monkeypatch.setattr(
        my_exoskeleton_flat,
        "_foot_contacts",
        lambda _env: (torch.tensor([0.0, 10.0, 0.0, 10.0]), torch.tensor([0.0, 0.0, 10.0, 10.0])),
    )

    reward = my_exoskeleton_flat.phase_contact_reward(FakeEnv(), gait=3, flight_window=0.35)

    assert torch.equal(reward, torch.tensor([1.0, 1.0, 1.0, 0.0]))


def test_w2s_contact_accepts_double_support_without_relaxing_walk(monkeypatch):
    """走转站（W2S）着地过程应渐进进入双脚支撑，无接触状态仍然无效。"""
    gait_modes = torch.full((3,), 2, dtype=torch.long)

    class FakeCommandTerm:
        gait_mode = gait_modes
        gait_phase = torch.tensor([torch.pi / 2.0, torch.pi / 2.0, torch.pi / 2.0])
        transition_time = torch.tensor([0.0, 1.5, 1.5])

    class FakeCommandManager:
        @staticmethod
        def get_term(_name):
            return FakeCommandTerm()

    class FakeEnv:
        command_manager = FakeCommandManager()

    monkeypatch.setattr(
        my_exoskeleton_flat,
        "_foot_contacts",
        lambda _env: (torch.tensor([10.0, 10.0, 0.0]), torch.tensor([0.0, 10.0, 0.0])),
    )

    reward = my_exoskeleton_flat.phase_contact_reward(
        FakeEnv(), gait=2, allow_double_support=True, double_support_ramp_s=1.5
    )

    assert torch.equal(reward, torch.tensor([1.0, 1.0, 0.0]))


def test_contact_reward_is_binary_instead_of_force_scaled(monkeypatch):
    """很大的冲击力不得产生无上限的正奖励。"""

    class FakeEnv:
        num_envs = 3

    monkeypatch.setattr(
        my_exoskeleton_flat,
        "_foot_contacts",
        lambda _env: (
            torch.tensor([0.0, 10.0, 3_000.0]),
            torch.tensor([0.0, 0.0, 0.0]),
        ),
    )

    result = my_exoskeleton_flat.contact_reward(FakeEnv())
    assert torch.equal(result, torch.tensor([0.0, 1.0, 1.0]))
    assert torch.all((result >= 0.0) & (result <= 1.0))


def test_sensor_stats_reads_supported_normal_force_api():
    """PhysX 接触特征不应使用只会发出警告的总力别名。"""

    class FakeData:
        net_normal_forces_w = torch.tensor(
            [
                [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 12.0], [0.0, 0.0, 4.0]],
            ]
        )

        @property
        def net_forces_w(self):
            raise AssertionError("unsupported total-force alias should not be accessed")

    class FakeSensor:
        data = FakeData()

    force_sum, force_mean, force_std = my_exoskeleton_flat._sensor_stats(
        FakeSensor(), env_count=2, device=torch.device("cpu")
    )

    assert torch.equal(force_sum, torch.tensor([5.0, 16.0]))
    assert torch.equal(force_mean, torch.tensor([2.5, 8.0]))
    assert torch.equal(force_std, torch.tensor([2.5, 4.0]))


def test_standing_feet_alignment_reward_is_limited_to_two_feet():
    """站立时的脚朝向项必须只读取左右两个脚部刚体。"""
    cfg = MyExoskeletonFlatEnvCfg()
    term = cfg.rewards.stand_feet_alignment

    assert term.weight == 0.5
    assert term.params["asset_cfg"].name == "robot"
    assert term.params["asset_cfg"].body_names == ["Lfoot", "Rfoot"]
    assert term.params["asset_cfg"].preserve_order is True
    assert term.params["sigma"] == 0.2


def test_exoskeleton_imus_support_selectable_kit_debug_probes():
    """所有 IMU 都应延迟加载自定义实现，并默认关闭训练探针。"""
    cfg = MyExoskeletonFlatEnvCfg()

    for sensor_name in ("imu_rthigh", "imu_rleg", "imu_rfoot", "imu_lthigh", "imu_lleg", "imu_lfoot"):
        sensor_cfg = getattr(cfg.scene, sensor_name)
        assert str(sensor_cfg.class_type) == "isaaclab_assets.sensors.inspectable_imu:InspectableImu"
        assert sensor_cfg.debug_vis is False


def test_feet_forward_alignment_reward_uses_yaw_and_straight_command_gate():
    """给定五种脚尖朝向与命令组合，只在直行命令下评价双脚相对机身的偏航角。

    第三行让机身和双脚一起旋转 0.5 rad，验证奖励比较的是相对朝向，而不是错误地要求
    世界坐标偏航角永远为零；最后两行验证横移或转向命令会关闭这项直行奖励。
    """

    def yaw_quat(yaw):
        return torch.stack(
            (
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                torch.sin(yaw / 2.0),
                torch.cos(yaw / 2.0),
            ),
            dim=-1,
        )

    root_yaw = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0])
    foot_yaw = torch.tensor(
        [
            [0.0, 0.0],
            [0.4, -0.4],
            [0.5, 0.5],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    commands = torch.tensor(
        [
            [0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.4, 0.1, 0.0],
            [0.4, 0.0, 0.1],
        ]
    )

    class FakeData:
        root_quat_w = yaw_quat(root_yaw)
        body_link_quat_w = yaw_quat(foot_yaw)

    class FakeRobot:
        data = FakeData()

    class FakeCommandManager:
        @staticmethod
        def get_command(name):
            assert name == "base_velocity"
            return commands

    class FakeEnv:
        num_envs = len(commands)
        scene = {"robot": FakeRobot()}
        command_manager = FakeCommandManager()

    reward = my_exoskeleton_flat.feet_forward_alignment_reward(
        FakeEnv(),
        asset_cfg=SceneEntityCfg("robot", body_ids=[0, 1]),
        command_name="base_velocity",
        sigma=0.2,
        min_forward_speed=0.05,
        max_lateral_speed=0.05,
        max_yaw_rate=0.05,
    )

    assert torch.isclose(reward[0], torch.tensor(1.0))
    assert reward[1] < reward[0]
    assert torch.isclose(reward[2], torch.tensor(1.0))
    assert torch.equal(reward[3:], torch.zeros(2))


def test_knee_limit_penalty_and_termination_respect_each_leg_sign():
    """给定正常弯曲与左右各一次反弯，惩罚和终止必须遵守两膝相反的关节符号。"""

    class FakeData:
        joint_pos = torch.tensor(
            [
                [-1.0, 1.0],  # 左右膝都按人体允许方向弯曲。
                [0.20, 1.0],  # 右膝符号为正，表示右腿反向弯曲。
                [-1.0, -0.20],  # 左膝符号为负，表示左腿反向弯曲。
            ]
        )
        joint_vel = torch.zeros(3, 2)

    class FakeRobot:
        data = FakeData()

    class FakeEnv:
        scene = {"robot": FakeRobot()}

    asset_cfg = SceneEntityCfg("robot", joint_ids=[0, 1], joint_names=list(KNEE_JOINT_NAMES), preserve_order=True)
    penalty = my_exoskeleton_flat.knee_limit_penalty(FakeEnv(), asset_cfg)
    terminated = my_exoskeleton_flat.knee_out_of_human_range(FakeEnv(), asset_cfg)

    assert torch.isclose(penalty[0], torch.tensor(0.0))
    assert penalty[1] > 0.0
    assert penalty[2] > 0.0
    assert torch.equal(terminated, torch.tensor([False, True, True]))


def test_human_knee_gait_reward_maps_flexion_per_side(monkeypatch):
    """左右镜像膝关节具有相同解剖屈曲角时，应得到相同分数。"""

    class FakeData:
        # ``q_R=-0.8`` 和 ``q_L=+0.8`` 都表示 0.8 rad 的解剖屈曲角。
        joint_pos = torch.tensor([[-0.8, 0.8], [0.0, 0.0]])

    class FakeRobot:
        data = FakeData()

    class FakeCommand:
        gait_mode = torch.ones(2, dtype=torch.long)

    class FakeCommandManager:
        @staticmethod
        def get_term(_name):
            return FakeCommand()

    class FakeEnv:
        scene = {"robot": FakeRobot()}
        command_manager = FakeCommandManager()

    # 第一行表示双腿摆动，第二行表示双腿支撑。
    monkeypatch.setattr(
        my_exoskeleton_flat,
        "_foot_contacts",
        lambda _env: (torch.tensor([0.0, 10.0]), torch.tensor([0.0, 10.0])),
    )
    reward = my_exoskeleton_flat.human_knee_gait_reward(
        FakeEnv(), gait=1, asset_cfg=SceneEntityCfg("robot", joint_ids=[0, 1], joint_names=list(KNEE_JOINT_NAMES))
    )

    assert torch.allclose(reward, torch.ones_like(reward), atol=1.0e-5)
