# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""外骨骼 ruleresult 轨迹导出契约的纯 Python 测试。

本文件保护24维字段顺序、单位、非均匀时间差分、目标与 measured 反馈分离、reset 后
历史隔离，以及 JSONL/55AA 两种落盘路径。二进制集成测试使用独立固件公式反解黄金帧，
避免编码器和测试共用同一实现而同时犯错；未来序列测试也明确区分运动学外推和真正的
外部预测。
"""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab_assets.tasks.protocol_55aa import (
    PROTOCOL_55AA_CONFIG_SCHEMA,
    PROTOCOL_55AA_FRAME_SIZE,
    PROTOCOL_55AA_JOINT_ORDER,
    decode_raw_frame,
)
from isaaclab_assets.tasks.robot_contract import ACTUATED_JOINT_NAMES
from isaaclab_assets.tasks.ruleresult import (
    RULERESULT_DIM,
    RuleresultRecorderMixin,
    build_ruleresult_record,
    derive_target_kinematics,
    extrapolate_ruleresult_sequence,
    flatten_ruleresult,
    predict_ruleresult_sequence,
)


def _protocol_recorder_env(output_path, protocol_config_path):
    """给定输出和标定路径，构造只实现记录器所需接口的最小两环境夹具。"""

    class Term:
        _joint_names = ACTUATED_JOINT_NAMES
        processed_actions = torch.arange(1, 9, dtype=torch.float32).unsqueeze(0) * 0.02

    class Actions:
        action = torch.zeros(1, 8)

        def __init__(self):
            self.term = Term()

        def get_term(self, _name):
            return self.term

    class Env(RuleresultRecorderMixin):
        def __init__(self):
            self.cfg = SimpleNamespace(
                ruleresult_enabled=True,
                ruleresult_interval=1,
                ruleresult_env_id=0,
                ruleresult_output_path=str(output_path),
                ruleresult_output_format="protocol_55aa",
                ruleresult_protocol_config_path=protocol_config_path,
                ruleresult_dt_s=0.02,
                log_dir=None,
            )
            data = SimpleNamespace(
                joint_pos=torch.zeros(1, 8),
                joint_vel=torch.zeros(1, 8),
                joint_acc=torch.zeros(1, 8),
            )
            self.action_manager = Actions()
            self.scene = {"robot": SimpleNamespace(joint_names=ACTUATED_JOINT_NAMES, data=data)}
            self.sim = SimpleNamespace(current_time=0.02)

    return Env()


def _write_protocol_config(path):
    """写出测试专用、身份标定且带宽松安全范围的可用 v2 协议配置。"""

    path.write_text(
        json.dumps(
            {
                "schema": PROTOCOL_55AA_CONFIG_SCHEMA,
                "example_only": False,
                "byte_order": "big",
                "length": "0x32",
                "limit_source": "trajectory",
                "motors": [
                    {
                        "slot": slot,
                        "joint_name": name,
                        "position_min_rad": -10.0,
                        "position_max_rad": 10.0,
                        "velocity_full_scale_rad_s": 20.0,
                        "acceleration_full_scale_rad_s2": 65.535,
                        "command_max_velocity_rad_s": 2.0,
                        "command_max_acceleration_rad_s2": 60.0,
                    }
                    for slot, name in enumerate(PROTOCOL_55AA_JOINT_ORDER, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_target_kinematics_are_field_major_and_use_control_dt():
    """有限差分应按文档约定的单位生成角速度和角加速度。"""

    first = torch.zeros(8)
    second = torch.ones(8) * 0.02
    first_velocity, first_acceleration = derive_target_kinematics(first)
    second_velocity, second_acceleration = derive_target_kinematics(second, first, first_velocity)

    assert torch.equal(first_velocity, torch.zeros(8))
    assert torch.equal(first_acceleration, torch.zeros(8))
    assert torch.allclose(second_velocity, torch.ones(8))
    assert torch.allclose(second_acceleration, torch.ones(8) * 50.0)

    flattened = flatten_ruleresult(second, second_velocity, second_acceleration)
    assert flattened.shape == (RULERESULT_DIM,)
    assert torch.equal(flattened[:8], second)
    assert torch.equal(flattened[8:16], second_velocity)
    assert torch.equal(flattened[16:], second_acceleration)


def test_ruleresult_record_contains_canonical_joints_and_units():
    """单行输出必须能安全序列化为 JSON，并且字段足以自描述。"""

    values = torch.arange(8, dtype=torch.float32)
    record = build_ruleresult_record(
        step=4,
        episode=1,
        env_id=0,
        timestamp_s=0.08,
        angle_rad=values,
        velocity_rad_s=values + 1.0,
        angular_acceleration_rad_s2=values + 2.0,
    )

    assert record["schema"] == "ruleresult/v1"
    assert record["joint_names"] == list(ACTUATED_JOINT_NAMES)
    assert record["angle_rad"] == values.tolist()
    assert record["velocity_rad_s"] == (values + 1.0).tolist()
    assert record["angular_acceleration_rad_s2"] == (values + 2.0).tolist()


def test_target_kinematics_reject_invalid_period_or_shape():
    with pytest.raises(ValueError, match="dt_s must be positive"):
        derive_target_kinematics(torch.zeros(8), dt_s=0.0)
    with pytest.raises(ValueError, match="expected 8 joint targets"):
        derive_target_kinematics(torch.zeros(7))


def test_prediction_sequence_uses_non_uniform_timestamps():
    """给定不均匀时间戳，每段速度和加速度必须使用自己的真实间隔，而非固定20 ms。

    第二帧与第一帧相差30 ms，第三帧再相差50 ms；这些刻意不同的间隔构成黄金结果，
    可以发现把实际落地输出错误地当成严格50 Hz的问题。
    """

    targets = torch.stack((torch.zeros(8), torch.ones(8) * 0.02, torch.ones(8) * 0.08))
    result = predict_ruleresult_sequence(targets, timestamps_s=torch.tensor([0.02, 0.05, 0.10]))

    assert torch.allclose(result["velocities"][0], torch.zeros(8))
    assert torch.allclose(result["velocities"][1], torch.ones(8) * (0.02 / 0.03))
    assert torch.allclose(result["velocities"][2], torch.ones(8) * (0.06 / 0.05))
    assert torch.allclose(result["accelerations"][1], torch.ones(8) * ((0.02 / 0.03) / 0.025))
    assert result["valid_mask"].tolist() == [False, True, True]


def test_constant_acceleration_extrapolation_has_explicit_horizon():
    result = extrapolate_ruleresult_sequence(
        torch.zeros(8),
        torch.ones(8),
        torch.ones(8) * 2.0,
        horizon_steps=3,
        dt_s=0.02,
    )

    assert result["angles"].shape == (3, 8)
    assert torch.allclose(result["velocities"][0], torch.ones(8) * 1.04)
    assert torch.allclose(result["accelerations"][2], torch.ones(8) * 2.0)


def test_ruleresult_record_rejects_non_finite_time_metadata():
    values = torch.zeros(8)
    with pytest.raises(ValueError, match="timestamp_s must be finite"):
        build_ruleresult_record(
            step=0,
            timestamp_s=float("nan"),
            angle_rad=values,
            velocity_rad_s=values,
            angular_acceleration_rad_s2=values,
        )
    with pytest.raises(ValueError, match="dt_s must be finite and positive"):
        build_ruleresult_record(
            step=0,
            timestamp_s=0.0,
            dt_s=0.0,
            angle_rad=values,
            velocity_rad_s=values,
            angular_acceleration_rad_s2=values,
        )


def test_recorder_writes_target_and_measured_values(tmp_path):
    """环境适配器无需启动 Kit 也能写出一行规范 JSONL 记录。"""

    class Data:
        joint_pos = torch.zeros(1, 8)
        joint_vel = torch.zeros(1, 8)
        joint_acc = torch.zeros(1, 8)

    class Robot:
        joint_names = ACTUATED_JOINT_NAMES
        data = Data()

    class Term:
        _joint_names = ACTUATED_JOINT_NAMES
        processed_actions = torch.ones(1, 8) * 0.1

    class Actions:
        action = torch.ones(1, 8) * 0.2

        @staticmethod
        def get_term(_name):
            return Term()

    class Cfg:
        ruleresult_enabled = True
        ruleresult_interval = 1
        ruleresult_env_id = 0
        ruleresult_output_path = str(tmp_path / "result.jsonl")
        ruleresult_dt_s = 0.02
        log_dir = None

    class Sim:
        current_time = 0.02

    class Env(RuleresultRecorderMixin):
        cfg = Cfg()
        action_manager = Actions()
        scene = {"robot": Robot()}
        sim = Sim()

    env = Env()
    env._ruleresult_record_after_step((None, None, torch.tensor([False]), torch.tensor([False])))
    assert env.ruleresult is not None
    assert env.ruleresult["angle_rad"] == pytest.approx([0.1] * 8)
    assert env.ruleresult["velocity_rad_s"] == pytest.approx([0.0] * 8)
    assert env.ruleresult_tensor.shape == (1, 24)
    assert (tmp_path / "result.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_recorder_writes_one_protocol_frame_and_keeps_readable_result(tmp_path):
    config_path = tmp_path / "protocol.json"
    output_path = tmp_path / "frames.bin"
    _write_protocol_config(config_path)
    env = _protocol_recorder_env(output_path, str(config_path))

    env._ruleresult_record_after_step((None, None, torch.tensor([False]), torch.tensor([False])))

    payload = output_path.read_bytes()
    decoded = decode_raw_frame(payload)
    assert len(payload) == PROTOCOL_55AA_FRAME_SIZE
    assert len(decoded.motors) == 8
    assert isinstance(env.ruleresult, dict)
    assert env.ruleresult["schema"] == "ruleresult/v1"
    assert env.ruleresult["joint_names"] == list(ACTUATED_JOINT_NAMES)


def test_recorder_frames_match_independent_firmware_decode_and_eight_motor_mapping(tmp_path):
    """连续记录两帧后，用独立固件公式验证8电机顺序、带符号速度和固定量程。"""

    config_path = tmp_path / "protocol.json"
    output_path = tmp_path / "frames.bin"
    _write_protocol_config(config_path)
    env = _protocol_recorder_env(output_path, str(config_path))
    step_result = (None, None, torch.tensor([False]), torch.tensor([False]))

    env._ruleresult_record_after_step(step_result)
    env.action_manager.term.processed_actions += torch.tensor(
        [[-0.004, 0.008, 0.012, -0.016, 0.020, -0.012, -0.008, 0.004]]
    )
    env.sim.current_time = 0.04
    env._ruleresult_record_after_step(step_result)

    frames = output_path.read_bytes()
    assert len(frames) == PROTOCOL_55AA_FRAME_SIZE * 2
    frame = frames[PROTOCOL_55AA_FRAME_SIZE:]
    record = env.ruleresult
    source_index = {name: index for index, name in enumerate(record["joint_names"])}
    for slot, name in enumerate(PROTOCOL_55AA_JOINT_ORDER):
        index = source_index[name]
        angle = record["angle_rad"][index]
        velocity = record["velocity_rad_s"][index]
        acceleration = abs(record["angular_acceleration_rad_s2"][index])
        offset = 3 + 6 * slot
        # 完全按照实体机 C 代码拼接两个大端字节，不复用 Python 解码器。
        raw_angle = (frame[offset] << 8) | frame[offset + 1]
        raw_velocity = (frame[offset + 2] << 8) | frame[offset + 3]
        raw_acceleration = (frame[offset + 4] << 8) | frame[offset + 5]
        assert raw_angle == math.floor(((angle + 12.57) / 25.14) * 65535.0 + 0.5)
        assert raw_velocity == math.floor(((velocity + 20.0) / 40.0) * 65535.0 + 0.5)
        assert raw_acceleration == math.floor(acceleration * 1000.0 + 0.5)
        assert raw_angle / 65535.0 * 25.14 - 12.57 == pytest.approx(angle, abs=0.000192)
        assert raw_velocity / 65535.0 * 40.0 - 20.0 == pytest.approx(velocity, abs=0.000306)
        assert raw_acceleration / 1000.0 == pytest.approx(acceleration, abs=0.0005)


def test_protocol_recorder_refuses_to_append_to_existing_nonempty_file(tmp_path):
    config_path = tmp_path / "protocol.json"
    output_path = tmp_path / "frames.bin"
    _write_protocol_config(config_path)
    previous_contents = b"previous protocol generation"
    output_path.write_bytes(previous_contents)
    env = _protocol_recorder_env(output_path, str(config_path))

    with pytest.raises(ValueError, match="refusing to append.*existing non-empty"):
        env._ruleresult_record_after_step((None, None, torch.tensor([False]), torch.tensor([False])))

    assert output_path.read_bytes() == previous_contents


def test_protocol_output_requires_a_configuration_path(tmp_path):
    env = _protocol_recorder_env(tmp_path / "frames.bin", None)

    with pytest.raises(ValueError, match="ruleresult_protocol_config_path is required"):
        env._ruleresult_record_after_step((None, None, torch.tensor([False]), torch.tensor([False])))


def test_recorder_uses_wall_dt_and_emits_optional_kinematic_prediction(tmp_path, monkeypatch):
    """墙钟模式应暴露真实调度抖动，但不能改变扁平的 24 维结果张量。"""

    import isaaclab_assets.tasks.ruleresult as ruleresult_module

    clock_values = iter((0.0, 0.02, 0.055))
    monkeypatch.setattr(ruleresult_module.time, "perf_counter", lambda: next(clock_values))

    class Data:
        joint_pos = torch.zeros(1, 8)
        joint_vel = torch.zeros(1, 8)
        joint_acc = torch.zeros(1, 8)

    class Robot:
        joint_names = ACTUATED_JOINT_NAMES
        data = Data()

    class Term:
        _joint_names = ACTUATED_JOINT_NAMES
        processed_actions = torch.zeros(1, 8)

    class Actions:
        def __init__(self):
            self.action = torch.zeros(1, 8)
            self.term = Term()

        def get_term(self, _name):
            return self.term

    class Cfg:
        ruleresult_enabled = True
        ruleresult_interval = 1
        ruleresult_env_id = 0
        ruleresult_output_path = str(tmp_path / "wall.jsonl")
        ruleresult_dt_s = 0.02
        ruleresult_time_mode = "wall_clock"
        ruleresult_min_dt_s = 1.0e-6
        ruleresult_max_dt_s = 1.0
        ruleresult_prediction_horizon = 2
        ruleresult_prediction_mode = "kinematic"
        ruleresult_prediction_dt_s = 0.02
        log_dir = None

    class Env(RuleresultRecorderMixin):
        cfg = Cfg()
        scene = {"robot": Robot()}

        def __init__(self):
            self.action_manager = Actions()

    env = Env()
    env.action_manager.get_term("joint_pos").processed_actions[:] = 0.1
    env._ruleresult_record_after_step((None, None, torch.tensor([False]), torch.tensor([False])))
    env.action_manager.get_term("joint_pos").processed_actions[:] = 0.135
    env._ruleresult_record_after_step((None, None, torch.tensor([False]), torch.tensor([False])))

    assert env.ruleresult["sample_dt_s"] == pytest.approx(0.035)
    assert env.ruleresult["velocity_rad_s"][0] == pytest.approx(1.0)
    assert env.ruleresult["dt_source"] == "wall_clock"
    assert env.ruleresult["prediction"]["source"] == "constant_acceleration_extrapolation"
    assert env.ruleresult["prediction"]["horizon_steps"] == 2


def test_recorder_keeps_terminal_frame_and_isolates_reset_history(tmp_path):
    """终止时先保存最后一帧，随后 reset 的新回合不能沿用上一回合的差分历史。

    两个并行环境分别通过 ``_reset_idx`` 和 ``reset_to`` 重置；未重置行仍应算出50 rad/s，
    被重置行的首帧速度必须归零，从而证明历史是逐环境隔离的。
    """

    class Data:
        def __init__(self):
            self.joint_pos = torch.zeros(2, 8)
            self.joint_vel = torch.zeros(2, 8)
            self.joint_acc = torch.zeros(2, 8)

    class Robot:
        joint_names = ACTUATED_JOINT_NAMES

        def __init__(self):
            self.data = Data()

    class Term:
        _joint_names = ACTUATED_JOINT_NAMES

        def __init__(self):
            self.processed_actions = torch.zeros(2, 8)

    class Actions:
        def __init__(self):
            self.term = Term()
            self.action = torch.zeros(2, 8)

        def get_term(self, _name):
            return self.term

    class Cfg:
        ruleresult_enabled = True
        ruleresult_interval = 1
        ruleresult_env_id = 0
        ruleresult_dt_s = 0.02
        log_dir = None

        def __init__(self, output_path):
            self.ruleresult_output_path = str(output_path)

    class Sim:
        current_time = 0.02

    class Base:
        def _reset_idx(self, env_ids):
            ids = torch.as_tensor(env_ids, dtype=torch.long)
            self.action_manager.term.processed_actions[ids] = 99.0
            self.scene["robot"].data.joint_pos[ids] = 99.0
            self.scene["robot"].data.joint_vel[ids] = 0.0
            self.scene["robot"].data.joint_acc[ids] = 0.0

        @staticmethod
        def reset_to(_state, _env_ids, seed=None, is_relative=False):
            del seed, is_relative
            return "restored"

    class Env(RuleresultRecorderMixin, Base):
        def __init__(self, output_path):
            self.cfg = Cfg(output_path)
            self.action_manager = Actions()
            self.scene = {"robot": Robot()}
            self.sim = Sim()

    env = Env(tmp_path / "terminal.jsonl")
    env.action_manager.term.processed_actions[:] = torch.tensor([[1.0] * 8, [10.0] * 8])
    env._ruleresult_record_after_step((None, None, torch.tensor([False, False]), torch.tensor([False, False])))

    env.action_manager.term.processed_actions[:] = torch.tensor([[2.0] * 8, [11.0] * 8])
    env.scene["robot"].data.joint_pos[0] = 1.5
    env._reset_idx([0])
    env._ruleresult_record_after_step((None, None, torch.tensor([True, False]), torch.tensor([False, False])))

    assert env.ruleresult["angle_rad"] == pytest.approx([2.0] * 8)
    assert env.ruleresult["measured"]["angle_rad"] == pytest.approx([1.5] * 8)
    assert env.ruleresult_tensor[0, 8] == pytest.approx(50.0)
    assert env.ruleresult_tensor[1, 8] == pytest.approx(50.0)

    env.action_manager.term.processed_actions[:] = torch.tensor([[100.0] * 8, [12.0] * 8])
    env._ruleresult_record_after_step((None, None, torch.tensor([False, False]), torch.tensor([False, False])))
    assert env.ruleresult_tensor[0, 8] == pytest.approx(0.0)
    assert env.ruleresult_tensor[1, 8] == pytest.approx(50.0)

    env.action_manager.term.processed_actions[:] = torch.tensor([[101.0] * 8, [13.0] * 8])
    assert env.reset_to({}, [1]) == "restored"
    env._ruleresult_record_after_step((None, None, torch.tensor([False, False]), torch.tensor([False, False])))
    assert env.ruleresult_tensor[0, 8] == pytest.approx(50.0)
    assert env.ruleresult_tensor[1, 8] == pytest.approx(0.0)
