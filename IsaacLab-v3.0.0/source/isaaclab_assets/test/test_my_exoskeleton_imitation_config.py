"""多模态模仿环境配置与教师目标序列的无 Kit 回归测试。

本文件保护4096环境默认场景、相机/IMU命名和固定观测 shape，并重点检查多个并行环境
共享缓存时的局部 reset：只重置一行不能截断其他环境仍在使用的序列，done 环境不能
继续推进旧目标，非周期任务到末尾必须保持最后一帧而不是突然跳回开头。
"""

from types import SimpleNamespace

import torch

from isaaclab_assets.envs.my_exoskeleton_imitation_env_cfg import MyExoskeletonImitationEnvCfg
from isaaclab_assets.tasks.my_exoskeleton_imitation import MyExoskeletonImitationEnv


def test_imitation_config_uses_fixed_4096_scene_and_camera_contract():
    cfg = MyExoskeletonImitationEnvCfg()

    assert cfg.scene.num_envs == 4096
    assert cfg.scene.channel_length_m == 2.4
    assert cfg.scene.left_rgb_camera.width == 256
    assert cfg.scene.left_rgb_camera.height == 256
    assert cfg.scene.right_rgb_camera.width == 256
    assert cfg.scene.right_rgb_camera.height == 256
    assert cfg.scene.left_rgb_camera.prim_path == "{ENV_REGEX_NS}/left_rgb_camera"
    assert cfg.scene.right_rgb_camera.prim_path == "{ENV_REGEX_NS}/right_rgb_camera"

    # 通道尺寸直接写成场景几何体，而不只是元数据属性。这样在 Viser/Newton 以及固定
    # 侧视相机画面中都能看见通道占用范围。
    assert cfg.scene.channel_floor.prim_path == "{ENV_REGEX_NS}/Channel/Floor"
    assert cfg.scene.channel_floor.spawn.size == (2.4, 2.4, 0.012)
    assert cfg.scene.channel_left_edge.prim_path == "{ENV_REGEX_NS}/Channel/LeftEdge"
    assert cfg.scene.channel_right_edge.prim_path == "{ENV_REGEX_NS}/Channel/RightEdge"
    assert cfg.scene.channel_left_edge.init_state.pos[1] > 1.2
    assert cfg.scene.channel_right_edge.init_state.pos[1] < -1.2
    assert cfg.scene.channel_floor.spawn.collision_props.collision_enabled is False


def test_imitation_config_exposes_six_recorder_imus_and_task_selector():
    cfg = MyExoskeletonImitationEnvCfg()

    assert [getattr(cfg.scene, f"imu_{index:02d}").prim_path for index in range(1, 7)] == [
        "{ENV_REGEX_NS}/Robot/Geometry/base_link/Lturn/Lthigh/Lleg/Lfoot",
        "{ENV_REGEX_NS}/Robot/Geometry/base_link/R_turn/Rthigh/Rleg/Rfoot",
        "{ENV_REGEX_NS}/Robot/Geometry/base_link/Lturn/Lthigh/Lleg",
        "{ENV_REGEX_NS}/Robot/Geometry/base_link/R_turn/Rthigh/Rleg",
        "{ENV_REGEX_NS}/Robot/Geometry/base_link/Lturn/Lthigh",
        "{ENV_REGEX_NS}/Robot/Geometry/base_link/R_turn/Rthigh",
    ]
    assert cfg.commands.base_velocity.task_id == "all"
    assert cfg.commands.base_velocity.random_task_sampling is True
    assert cfg.imitation_window_size == 32


def test_optional_sensor_switches_default_on_and_preserve_shape_contract():
    """传感器开关默认启用，而且不得改变已经声明的观测空间。"""

    cfg = MyExoskeletonImitationEnvCfg()
    assert cfg.use_imu is True
    assert cfg.use_rgb is True
    assert cfg.use_event is True
    assert cfg.use_contact is True
    full_space = cfg.observation_space

    disabled = MyExoskeletonImitationEnvCfg(use_imu=False, use_rgb=False, use_event=False, use_contact=False)
    assert disabled.use_imu is False
    assert disabled.use_rgb is False
    assert disabled.use_event is False
    assert disabled.use_contact is False
    assert disabled.observation_space == full_space
    assert disabled.imitation_rgb_features is False
    assert disabled.imitation_event_features is False


def _make_sequence_test_env() -> SimpleNamespace:
    """构造只含目标序列字段的两行假环境，用固定值隔离检查缓冲区索引逻辑。

    序列缓存填 9、当前目标填 99，使测试能一眼判断某一行是被正确读取、保持不变，
    还是在局部 reset 时被另一环境意外覆盖。
    """

    class Cfg:
        imitation_random_task_sampling = False
        imitation_task_id = "BT02"

    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), cfg=Cfg())
    env._imitation_task_ids = ["BT02"]
    env._target_by_task = {}
    env._imitation_task_ids_per_env = []
    env._imitation_sequence_lengths = torch.full((2,), 4, dtype=torch.long)
    env._imitation_target_cursor = torch.zeros(2, dtype=torch.long)
    env._imitation_sequence_targets = {
        "target_imu": torch.full((2, 4, 6, 9), 9.0),
        "target_motion": torch.full((2, 4, 8), 9.0),
        "target_velocity": torch.full((2, 4, 2), 9.0),
        "target_rgb_feature": torch.full((2, 4, 12), 9.0),
        "target_event_feature": torch.full((2, 4, 32), 9.0),
    }
    env._imitation_targets = {
        "target_imu": torch.zeros(2, 6, 9),
        "imu_valid": torch.zeros(2, 6, dtype=torch.bool),
        "target_motion": torch.full((2, 8), 99.0),
        "motion_valid": torch.zeros(2, dtype=torch.bool),
        "target_velocity": torch.zeros(2, 2),
        "velocity_valid": torch.zeros(2, dtype=torch.bool),
        "target_rgb_feature": torch.zeros(2, 12),
        "rgb_valid": torch.zeros(2, dtype=torch.bool),
        "target_event_feature": torch.zeros(2, 32),
        "event_valid": torch.zeros(2, dtype=torch.bool),
    }
    env._write_current_targets = lambda env_ids=None: MyExoskeletonImitationEnv._write_current_targets(env, env_ids)
    return env


def test_target_sequence_partial_reset_preserves_other_environment_buffer():
    """只重置环境0并装入一帧短序列时，环境1的四帧缓存和当前目标必须原样保留。"""

    env = _make_sequence_test_env()
    env._select_task_ids = lambda env_ids: ["BT02" for _ in range(env_ids.numel())]

    def short_target(task_id):
        return {
            "target_imu": torch.full((1, 6, 9), 7.0),
            "target_motion": torch.full((1, 8), 7.0),
            "target_velocity": torch.full((1, 2), 7.0),
            "target_rgb_feature": torch.full((1, 12), 7.0),
            "target_event_feature": torch.full((1, 32), 7.0),
            "imu_valid": torch.ones(6, dtype=torch.bool),
            "motion_valid": torch.tensor(True),
            "velocity_valid": torch.tensor(True),
            "rgb_valid": torch.tensor(True),
            "event_valid": torch.tensor(True),
            "sequence_length": 1,
            "task_id": task_id,
        }

    env._target_for_task = short_target
    MyExoskeletonImitationEnv._sample_targets(env, torch.tensor([0]))

    assert env._imitation_sequence_targets["target_motion"].shape[1] == 4
    assert torch.equal(env._imitation_sequence_targets["target_motion"][1], torch.full((4, 8), 9.0))
    assert torch.equal(env._imitation_targets["target_motion"][1], torch.full((8,), 99.0))


def test_target_sequence_advance_skips_done_environments():
    """给定环境1已经 done，推进序列时只能移动环境0的游标和当前目标。"""

    env = _make_sequence_test_env()
    env._imitation_target_cursor[:] = torch.tensor([0, 2])
    env._imitation_sequence_lengths[:] = torch.tensor([4, 4])
    MyExoskeletonImitationEnv._advance_target_sequences(env, torch.tensor([False, True]))

    assert torch.equal(env._imitation_target_cursor, torch.tensor([1, 2]))
    assert torch.equal(env._imitation_targets["target_motion"][0], torch.full((8,), 9.0))
    assert torch.equal(env._imitation_targets["target_motion"][1], torch.full((8,), 99.0))


def test_non_cyclic_target_sequence_holds_final_frame():
    """过渡动作记录播放到终点后，不得从静止姿态突然跳回起始姿态。"""

    env = _make_sequence_test_env()
    env._imitation_sequence_loop_per_env = torch.tensor([False, True])
    env._imitation_target_cursor[:] = torch.tensor([3, 3])
    env._imitation_sequence_lengths[:] = torch.tensor([4, 4])

    MyExoskeletonImitationEnv._advance_target_sequences(env)

    assert torch.equal(env._imitation_target_cursor, torch.tensor([3, 0]))
