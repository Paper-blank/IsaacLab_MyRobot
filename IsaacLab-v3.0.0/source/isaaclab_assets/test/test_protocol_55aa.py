# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""8电机 ``55 AA`` 线路协议的纯 Python 合约测试。

这些测试用独立黄金字节保护53B帧布局、大端字段、XOR范围、槽位1至8顺序、流式重新
同步和固件物理量公式。还验证量化前与反解后的双重软限位、v1/示例配置拒绝，以及
方向和减速比只应用一次。该测试不连接真实总线，但可阻止会让固件误读电机命令的字节
级回归。
"""

import math

import pytest

from isaaclab_assets.tasks.protocol_55aa import (
    PROTOCOL_55AA_CONFIG_SCHEMA,
    PROTOCOL_55AA_FRAME_SIZE,
    PROTOCOL_55AA_JOINT_ORDER,
    PROTOCOL_55AA_LENGTH,
    MotorQuantization,
    Protocol55AAConfig,
    Protocol55AAError,
    Protocol55AAStreamDecoder,
    RawMotorCommand,
    decode_raw_frame,
    encode_raw_frame,
)
from isaaclab_assets.tasks.robot_contract import ACTUATED_JOINT_NAMES


def _raw_commands() -> tuple[RawMotorCommand, ...]:
    """给定槽位1至8，生成每个 uint16 都不同的黄金原始电机记录。"""

    return (
        RawMotorCommand(0x0000, 0x0001, 0x0002),
        RawMotorCommand(0x0102, 0x0304, 0x0506),
        RawMotorCommand(0x1234, 0x5678, 0x9ABC),
        RawMotorCommand(0xFFFF, 0x8000, 0x7FFF),
        RawMotorCommand(0x1111, 0x2222, 0x3333),
        RawMotorCommand(0xA1B2, 0xC3D4, 0xE5F6),
        RawMotorCommand(0x1357, 0x2468, 0xABCD),
        RawMotorCommand(0x0F0E, 0x0D0C, 0x0B0A),
    )


def _motor(slot: int, joint_name: str, **overrides) -> MotorQuantization:
    """生成一个默认宽松、可按测试覆盖单字段的电机标定夹具。"""

    values = {
        "slot": slot,
        "joint_name": joint_name,
        "position_min_rad": -1.0,
        "position_max_rad": 1.0,
        "velocity_full_scale_rad_s": 20.0,
        "acceleration_full_scale_rad_s2": 65.535,
        "limit_source": "trajectory",
    }
    return MotorQuantization(**(values | overrides))


def _configuration_payload() -> dict:
    """生成可部署标志已关闭、8槽位齐全的最小 v2 配置对象。"""

    return {
        "schema": PROTOCOL_55AA_CONFIG_SCHEMA,
        "byte_order": "big",
        "length": "0x32",
        "example_only": False,
        "limit_source": "trajectory",
        "motors": [
            {
                "slot": slot,
                "joint_name": name,
                "position_min_rad": -1.0,
                "position_max_rad": 1.0,
                "velocity_full_scale_rad_s": 20.0,
                "acceleration_full_scale_rad_s2": 65.535,
            }
            for slot, name in enumerate(PROTOCOL_55AA_JOINT_ORDER, start=1)
        ],
    }


def test_zero_frame_has_fixed_size_length_and_checksum():
    """给定8个全零电机槽位，帧长、长度字节、48B负载、校验和帧尾应全部固定。"""

    frame = encode_raw_frame((RawMotorCommand(0, 0, 0),) * 8)

    assert PROTOCOL_55AA_FRAME_SIZE == 53
    assert PROTOCOL_55AA_LENGTH == 0x32
    assert len(frame) == 53
    assert frame[:3] == bytes.fromhex("55 AA 32")
    assert frame[3:51] == bytes(48)
    assert frame[-2:] == bytes.fromhex("CD 5A")


def test_raw_frame_uses_fixed_motor_slots_and_firmware_big_endian_fields():
    """给定人工编写的 uint16 黄金值，48B 负载必须按槽位1至8和大端顺序逐字节匹配。"""

    commands = _raw_commands()
    frame = encode_raw_frame(commands)
    expected_payload = bytes.fromhex(
        "00 00 00 01 00 02 "
        "01 02 03 04 05 06 "
        "12 34 56 78 9A BC "
        "FF FF 80 00 7F FF "
        "11 11 22 22 33 33 "
        "A1 B2 C3 D4 E5 F6 "
        "13 57 24 68 AB CD "
        "0F 0E 0D 0C 0B 0A"
    )

    assert frame[3:51] == expected_payload
    assert decode_raw_frame(frame).motors == commands


def test_protocol_fixes_joint_names_to_slots_one_through_eight():
    assert PROTOCOL_55AA_JOINT_ORDER == (
        "Lleg_to_Lfoot",
        "Rleg_to_Rfoot",
        "Lthigh_to_Lleg",
        "Rthigh_to_Rleg",
        "Lturn_to_Lthigh",
        "Rturn_to_Rthigh",
        "base_to_Lturn",
        "base_to_Rturn",
    )
    config = Protocol55AAConfig(
        tuple(_motor(slot, name) for slot, name in enumerate(PROTOCOL_55AA_JOINT_ORDER, start=1))
    )
    assert config.joint_names == PROTOCOL_55AA_JOINT_ORDER


def test_config_encode_reorders_model_joint_values_into_fixed_motor_slots():
    config = Protocol55AAConfig(
        tuple(_motor(slot, name) for slot, name in enumerate(PROTOCOL_55AA_JOINT_ORDER, start=1))
    )
    source_angles = tuple(-0.8 + 0.2 * index for index in range(8))
    angle_by_name = dict(zip(ACTUATED_JOINT_NAMES, source_angles))

    frame = config.encode(
        ACTUATED_JOINT_NAMES,
        source_angles,
        (0.0,) * 8,
        (0.0,) * 8,
    )
    decoded = decode_raw_frame(frame)
    expected_positions = tuple(
        math.floor(((angle_by_name[name] + 12.57) / 25.14) * 0xFFFF + 0.5) for name in PROTOCOL_55AA_JOINT_ORDER
    )

    assert tuple(motor.target_position for motor in decoded.motors) == expected_positions
    assert tuple(motor.target_velocity for motor in decoded.motors) == (32768,) * 8
    assert tuple(motor.max_acceleration for motor in decoded.motors) == (0,) * 8


def test_example_only_configuration_is_rejected_before_hardware_encoding():
    payload = _configuration_payload()
    payload["example_only"] = True

    with pytest.raises(Protocol55AAError, match="example"):
        Protocol55AAConfig.from_mapping(payload)


def test_configuration_schema_version_and_explicit_big_byte_order():
    assert PROTOCOL_55AA_CONFIG_SCHEMA == "exoskeleton-55aa-config/v2"
    assert Protocol55AAConfig.from_mapping(_configuration_payload()).joint_names == PROTOCOL_55AA_JOINT_ORDER


def test_new_configuration_defaults_to_trajectory_derivatives():
    payload = _configuration_payload()
    del payload["limit_source"]
    config = Protocol55AAConfig.from_mapping(payload)

    assert all(motor.limit_source == "trajectory" for motor in config.motors)
    frame = config.encode(PROTOCOL_55AA_JOINT_ORDER, (0.0,) * 8, (-1.0,) * 8, (1.0,) * 8)
    assert all(motor.max_speed < 32768 and motor.max_acceleration == 1000 for motor in decode_raw_frame(frame).motors)


@pytest.mark.parametrize(
    ("field", "value"),
    (("schema", "exoskeleton-55aa-config/v1"), ("byte_order", "little")),
)
def test_configuration_rejects_incompatible_v1_or_little_endian(field, value):
    payload = _configuration_payload()
    payload[field] = value

    with pytest.raises(Protocol55AAError):
        Protocol55AAConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("mutate", "error"),
    (
        (lambda frame: frame.__setitem__(0, 0x54), "header"),
        (lambda frame: frame.__setitem__(2, 0x31), "length"),
        (lambda frame: frame.__setitem__(-1, 0x00), "tail"),
        (lambda frame: frame.__setitem__(-2, frame[-2] ^ 0x01), "checksum"),
    ),
)
def test_decoder_rejects_corrupt_frame_fields(mutate, error):
    frame = bytearray(encode_raw_frame(_raw_commands()))
    mutate(frame)

    with pytest.raises(Protocol55AAError, match=error):
        decode_raw_frame(frame)


@pytest.mark.parametrize("removed_bytes", (1, 2, 17, 52))
def test_decoder_rejects_truncated_frames(removed_bytes):
    frame = encode_raw_frame(_raw_commands())

    with pytest.raises(Protocol55AAError, match="53 bytes"):
        decode_raw_frame(frame[:-removed_bytes])


def test_payload_may_contain_header_and_tail_byte_sequences():
    commands = list(_raw_commands())
    commands[0] = RawMotorCommand(0x55AA, 0x5A5A, 0x3255)
    frame = encode_raw_frame(commands)

    assert frame[3:5] == bytes.fromhex("55 AA")
    assert bytes.fromhex("5A 5A") in frame[3:51]
    decoder = Protocol55AAStreamDecoder()
    assert decoder.feed(frame)[0].motors == tuple(commands)


def test_stream_decoder_resynchronizes_after_noise_and_corrupt_frame():
    """给定前导噪声和坏校验帧，流解码器应保留半个帧头并在下一帧恢复同步。"""

    first_commands = _raw_commands()
    second_commands = tuple(reversed(first_commands))
    first = encode_raw_frame(first_commands)
    corrupt = bytearray(first)
    corrupt[-2] ^= 0x80
    second = encode_raw_frame(second_commands)
    decoder = Protocol55AAStreamDecoder()

    assert decoder.feed(b"\x00\xaa\x13" + bytes(corrupt) + b"\x55") == []
    assert decoder.buffered_bytes == 1
    assert decoder.feed(second[1:19]) == []
    decoded = decoder.feed(second[19:])

    assert [frame.motors for frame in decoded] == [second_commands]
    assert decoder.buffered_bytes == 0


@pytest.mark.parametrize(
    ("angle", "velocity", "acceleration"),
    (
        (math.nan, 0.0, 0.0),
        (0.0, math.inf, 0.0),
        (0.0, 0.0, -math.inf),
    ),
)
def test_quantization_rejects_non_finite_targets(angle, velocity, acceleration):
    with pytest.raises(Protocol55AAError, match="finite"):
        _motor(1, PROTOCOL_55AA_JOINT_ORDER[0]).encode(angle, velocity, acceleration)


@pytest.mark.parametrize(
    ("angle", "velocity", "acceleration", "error"),
    (
        (1.0001, 0.0, 0.0, "motor position"),
        (-1.0001, 0.0, 0.0, "motor position"),
        (0.0, 20.0001, 0.0, "velocity"),
        (0.0, -20.0001, 0.0, "velocity"),
        (0.0, 0.0, -65.5351, "acceleration"),
        (0.0, 0.0, 65.5351, "acceleration"),
    ),
)
def test_quantization_rejects_targets_outside_configured_ranges(angle, velocity, acceleration, error):
    with pytest.raises(Protocol55AAError, match=error):
        _motor(1, PROTOCOL_55AA_JOINT_ORDER[0]).encode(angle, velocity, acceleration)


def test_configuration_rejects_joint_names_in_the_wrong_motor_slots():
    wrong_order = list(PROTOCOL_55AA_JOINT_ORDER)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]

    with pytest.raises(Protocol55AAError, match="motor slot order"):
        Protocol55AAConfig(tuple(_motor(slot, name) for slot, name in enumerate(wrong_order, start=1)))


@pytest.mark.parametrize(
    ("angle", "velocity", "acceleration", "expected"),
    (
        (-12.57, -20.0, 0.0, (0, 0, 0)),
        (0.0, 0.0, 0.0, (32768, 32768, 0)),
        (12.57, 20.0, 65.535, (65535, 65535, 65535)),
        (0.0, 0.0, -65.535, (32768, 32768, 65535)),
    ),
)
def test_firmware_physical_minimum_zero_and_maximum_quantization(angle, velocity, acceleration, expected):
    """固件量化公式的最小值、中点、最大值和负加速度幅值必须命中黄金端点。"""

    motor = _motor(
        1,
        PROTOCOL_55AA_JOINT_ORDER[0],
        position_min_rad=-12.57,
        position_max_rad=12.57,
    )

    raw = motor.encode(angle, velocity, acceleration)

    assert (raw.target_position, raw.target_velocity, raw.acceleration_limit) == expected
    assert raw.max_speed == raw.target_velocity
    assert raw.max_acceleration == raw.acceleration_limit
    assert motor.decode_position_rad(raw.target_position) == pytest.approx(angle, abs=25.14 / 65535 / 2 + 1.0e-12)
    assert motor.decode_velocity_rad_s(raw.target_velocity) == pytest.approx(velocity, abs=40.0 / 65535 / 2 + 1.0e-12)
    assert motor.decode_acceleration_rad_s2(raw.acceleration_limit) == pytest.approx(abs(acceleration), abs=0.0005)


@pytest.mark.parametrize("joint_velocity", (-4.0, 4.0))
def test_quantization_preserves_velocity_sign_after_motor_direction_and_gear_ratio(joint_velocity):
    motor = _motor(
        1,
        PROTOCOL_55AA_JOINT_ORDER[0],
        zero_offset_rad=0.1,
        direction=-1,
        gear_ratio=2.0,
    )

    raw = motor.encode(0.2, joint_velocity, -0.5)
    expected_position = math.floor(((-0.3 + 12.57) / 25.14) * 65535 + 0.5)
    expected_velocity = math.floor(((-2.0 * joint_velocity + 20.0) / 40.0) * 65535 + 0.5)

    assert raw.target_position == expected_position
    assert raw.target_velocity == expected_velocity
    assert raw.acceleration_limit == 1000


def test_one_radian_per_second_squared_uses_fixed_thousand_scale_and_big_endian_bytes():
    motor = _motor(1, PROTOCOL_55AA_JOINT_ORDER[0])
    command = motor.encode(0.0, 0.0, 1.0)
    frame = encode_raw_frame((command,) * 8)

    assert command.acceleration_limit == 1000
    assert frame[7:9] == bytes.fromhex("03 E8")
    # 这里直接照抄固件 C 接收公式独立反解，不调用 Python codec，避免两边同时写错仍通过测试。
    raw_angle = (frame[3] << 8) | frame[4]
    raw_velocity = (frame[5] << 8) | frame[6]
    raw_acceleration = (frame[7] << 8) | frame[8]
    assert (raw_angle / 65535.0) * 25.14 - 12.57 == pytest.approx(0.0, abs=0.000192)
    assert (raw_velocity / 65535.0) * 40.0 - 20.0 == pytest.approx(0.0, abs=0.000306)
    assert raw_acceleration / 1000.0 == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("position_min_rad", -12.5701),
        ("position_max_rad", 12.5701),
        ("velocity_full_scale_rad_s", 10.0),
        ("velocity_full_scale_rad_s", 40.0),
        ("acceleration_full_scale_rad_s2", 20.0),
        ("acceleration_full_scale_rad_s2", 1000.0),
    ),
)
def test_configuration_cannot_change_firmware_wire_ranges(field, value):
    with pytest.raises(Protocol55AAError):
        _motor(1, PROTOCOL_55AA_JOINT_ORDER[0], **{field: value})


@pytest.mark.parametrize("limit_source", ("trajectory", "configured"))
@pytest.mark.parametrize(
    ("angle", "velocity", "acceleration"),
    ((1.0001, 0.0, 0.0), (0.0, 1.0001, 0.0), (0.0, -1.0001, 0.0), (0.0, 0.0, -2.0001)),
)
def test_stricter_motor_safety_limits_reject_trajectory_before_encoding(limit_source, angle, velocity, acceleration):
    motor = _motor(
        1,
        PROTOCOL_55AA_JOINT_ORDER[0],
        limit_source=limit_source,
        command_max_velocity_rad_s=1.0,
        command_max_acceleration_rad_s2=2.0,
    )

    with pytest.raises(Protocol55AAError):
        motor.encode(angle, velocity, acceleration)


@pytest.mark.parametrize(("limit_source", "expected_acceleration"), (("trajectory", 1000), ("configured", 2000)))
def test_configured_mode_only_substitutes_acceleration_and_keeps_signed_velocity(limit_source, expected_acceleration):
    motor = _motor(
        1,
        PROTOCOL_55AA_JOINT_ORDER[0],
        limit_source=limit_source,
        command_max_velocity_rad_s=1.0,
        command_max_acceleration_rad_s2=2.0,
    )

    raw = motor.encode(0.0, -0.5, 1.0)

    assert raw.target_velocity == math.floor(((20.0 - 0.5) / 40.0) * 65535 + 0.5)
    assert raw.acceleration_limit == expected_acceleration


@pytest.mark.parametrize(
    ("overrides", "velocity", "error"),
    (
        ({"command_max_velocity_rad_s": 1.0}, 1.0, "quantized target velocity"),
        ({"position_max_rad": 0.0}, 0.0, "quantized motor position"),
        (
            {
                "limit_source": "configured",
                "command_max_velocity_rad_s": 1.0,
                "command_max_acceleration_rad_s2": 2.0006,
            },
            0.0,
            "quantized acceleration magnitude",
        ),
        ({"command_max_velocity_rad_s": 0.0}, 0.0, "quantized target velocity"),
    ),
)
def test_quantization_rejects_rounding_that_crosses_a_motor_safety_limit(overrides, velocity, error):
    """输入值虽在软限位内，但量化再反解后越界时仍必须拒绝发送。

    这是浮点数映射到 uint16 时容易漏掉的第二道检查：真实电机执行的是固件反解值，
    而不是量化前那个看似安全的 Python 浮点数。
    """

    motor = _motor(1, PROTOCOL_55AA_JOINT_ORDER[0], **overrides)

    with pytest.raises(Protocol55AAError, match=error):
        motor.encode(0.0, velocity, 0.0)


@pytest.mark.parametrize("limit_source", ("trajectory", "configured"))
def test_quantization_accepts_targets_with_margin_inside_tighter_safety_bounds(limit_source):
    motor = _motor(
        1,
        PROTOCOL_55AA_JOINT_ORDER[0],
        position_min_rad=-0.5,
        position_max_rad=0.5,
        limit_source=limit_source,
        command_max_velocity_rad_s=1.0,
        command_max_acceleration_rad_s2=2.0,
    )

    raw = motor.encode(0.49, 0.999, 1.999)

    assert -0.5 <= motor.decode_position_rad(raw.target_position) <= 0.5
    assert -1.0 <= motor.decode_velocity_rad_s(raw.target_velocity) <= 1.0
    assert 0.0 <= motor.decode_acceleration_rad_s2(raw.acceleration_limit) <= 2.0


@pytest.mark.parametrize("factory", (MotorQuantization, MotorQuantization.from_mapping))
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("slot", 1.9),
        ("slot", True),
        ("slot", 1.0),
        ("slot", "1"),
        ("direction", 1.5),
        ("direction", True),
        ("direction", 1.0),
        ("direction", "1"),
        ("joint_name", None),
        ("joint_name", 1),
        ("joint_name", ""),
    ),
)
def test_dataclass_and_json_mapping_reject_ambiguous_motor_identity_values(factory, field, value):
    values = _configuration_payload()["motors"][0] | {field: value}

    with pytest.raises(Protocol55AAError):
        if factory is MotorQuantization:
            factory(**values)
        else:
            factory(values)
