# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""编码和校验外骨骼 8 电机 55 AA 控制帧。

线路格式有意不依赖 Isaac Sim 和 Torch，所以回放记录器、工控机程序与单元测试可以
复用同一实现。只有加载每个电机的明确标定参数后，物理量才会转换成 uint16。v2 与
固件保持一致：大端字节序、带偏移的有符号角度/速度，以及无符号加速度幅值；帧中
本身不携带协议版本号。

新手阅读顺序：先看顶部帧常量和固定槽位，再看 encode_raw_frame 的 53 字节结构，
最后看 MotorQuantization 如何应用零位、方向、减速比和软限位。本模块只负责
“物理量怎样变成确定字节”，不会自行打开 CAN、EtherCAT 或串口。

协议量程不等于人体或机械安全限位。编码器会在量化前检查原始物理量，并按固件公式
反解量化整数后再次检查；任何越界都拒绝，不做静默裁剪。帧本身没有版本号，所以旧
v1 二进制不能靠帧长或 XOR 自动识别，必须隔离并重新导出。
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_55AA_CONFIG_SCHEMA = "exoskeleton-55aa-config/v2"
PROTOCOL_55AA_BYTE_ORDER = "big"
PROTOCOL_55AA_HEADER = bytes((0x55, 0xAA))
PROTOCOL_55AA_LENGTH = 0x32
PROTOCOL_55AA_TAIL = 0x5A
PROTOCOL_55AA_MOTOR_COUNT = 8
PROTOCOL_55AA_MOTOR_SIZE = 6
PROTOCOL_55AA_PAYLOAD_SIZE = PROTOCOL_55AA_MOTOR_COUNT * PROTOCOL_55AA_MOTOR_SIZE
PROTOCOL_55AA_FRAME_SIZE = 2 + 1 + PROTOCOL_55AA_PAYLOAD_SIZE + 1 + 1
UINT16_MAX = 0xFFFF
PROTOCOL_55AA_POSITION_MIN_RAD = -12.57
PROTOCOL_55AA_POSITION_MAX_RAD = 12.57
PROTOCOL_55AA_VELOCITY_MIN_RAD_S = -20.0
PROTOCOL_55AA_VELOCITY_MAX_RAD_S = 20.0
PROTOCOL_55AA_ACCELERATION_MAX_RAD_S2 = 65.535
PROTOCOL_55AA_ACCELERATION_RESOLUTION_RAD_S2 = 0.001

# 一帧固定 53B，不能只看 length=0x32 就误认为总长 50B：
# [0..1] 55 AA 帧头；[2] 0x32；[3..50] 8 电机 * 6B；[51] XOR；[52] 5A。
# length 表示“长度字节之后的剩余字节数”，即 payload 48B + 校验 1B + 帧尾 1B。
# 每个电机 6B 又依次是目标位置、带偏置的有符号目标速度、非负加速度幅值/上限，
# 每项都是高字节在前的 uint16。此协议层不包含时间戳或电机反馈。

# 下面的元组是“线路槽位顺序”，不是 actor 的右腿优先顺序。编码时必须按关节名称
# 重排，不能把 8 维动作数组直接复制到 48 字节 payload。
PROTOCOL_55AA_JOINT_ORDER = (
    "Lleg_to_Lfoot",
    "Rleg_to_Rfoot",
    "Lthigh_to_Lleg",
    "Rthigh_to_Rleg",
    "Lturn_to_Lthigh",
    "Rturn_to_Rthigh",
    "base_to_Lturn",
    "base_to_Rturn",
)
"""固定槽位：左右踝、左右膝、左右髋屈伸、左右转向关节。"""


class Protocol55AAError(ValueError):
    """帧结构、校验、标定或量化配置不满足协议时抛出的异常。"""


def xor_checksum(data: bytes | bytearray | memoryview) -> int:
    """逐字节异或并返回 0..255 的校验值。

    完整帧的校验范围是帧头、长度和 48B payload，不包括校验字节本身与帧尾。
    """

    checksum = 0
    for value in data:
        checksum ^= int(value)
    return checksum


@dataclass(frozen=True)
class RawMotorCommand:
    """单个电机的 3 个原始 uint16 线路码，不是带单位的物理量。

    每个字段必须是 0..65535 的整数。``max_speed`` 是旧 API 名；v2 在线路上实际表示
    带 20 rad/s 偏置的“有符号目标速度”，新代码应通过 ``target_velocity`` 理解它。
    ``max_acceleration`` 仍是无符号加速度幅值/上限，绝不表示负加速度方向。
    """

    target_position: int
    max_speed: int
    max_acceleration: int

    def __post_init__(self) -> None:
        """拒绝 bool、浮点数和 uint16 范围外的线路字段。"""

        for name in ("target_position", "max_speed", "max_acceleration"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT16_MAX:
                raise Protocol55AAError(f"{name} must be an integer in [0, 65535], got {value!r}")

    @property
    def target_velocity(self) -> int:
        """返回带偏置的有符号速度码；这是旧字段 ``max_speed`` 的语义别名。"""

        return self.max_speed

    @property
    def acceleration_limit(self) -> int:
        """返回无符号加速度幅值/上限码；是 ``max_acceleration`` 的语义别名。"""

        return self.max_acceleration


@dataclass(frozen=True)
class Decoded55AAFrame:
    """已经通过帧头、长度、帧尾和 XOR 校验的8电机原始帧。"""

    motors: tuple[RawMotorCommand, ...]
    checksum: int

    def __post_init__(self) -> None:
        """确保解码结果恰好含8个电机记录。"""

        if len(self.motors) != PROTOCOL_55AA_MOTOR_COUNT:
            raise Protocol55AAError(f"a frame must contain {PROTOCOL_55AA_MOTOR_COUNT} motor records")


def encode_raw_frame(motors: Sequence[RawMotorCommand]) -> bytes:
    """把按槽位1到8排列的原始电机记录编码为一帧固定 53B 字节串。

    这里不做 rad 等物理量换算，只打包已经量化好的 uint16。全部 16 位字段使用大端序
    ``高字节, 低字节``，与固件 ``(high << 8) | low`` 一致。XOR 覆盖帧头、长度和
    48B 数据区，不覆盖 XOR 自身和帧尾。
    """

    records = tuple(motors)
    if len(records) != PROTOCOL_55AA_MOTOR_COUNT:
        raise Protocol55AAError(f"expected {PROTOCOL_55AA_MOTOR_COUNT} motor records, got {len(records)}")
    if not all(isinstance(record, RawMotorCommand) for record in records):
        raise Protocol55AAError("all motor records must be RawMotorCommand instances")
    payload = b"".join(
        struct.pack(">HHH", record.target_position, record.target_velocity, record.acceleration_limit)
        for record in records
    )
    prefix = PROTOCOL_55AA_HEADER + bytes((PROTOCOL_55AA_LENGTH,)) + payload
    frame = prefix + bytes((xor_checksum(prefix), PROTOCOL_55AA_TAIL))
    if len(frame) != PROTOCOL_55AA_FRAME_SIZE:
        raise AssertionError("internal 55AA frame-size mismatch")
    return frame


def decode_raw_frame(frame: bytes | bytearray | memoryview) -> Decoded55AAFrame:
    """严格校验一帧完整 53B 数据，并还原8组原始 uint16 字段。

    输入必须正好是一帧；CAN FD 的 64B DLC 填充或串口中相邻多帧不能直接传入，应由
    传输适配层先去掉填充，或交给 :class:`Protocol55AAStreamDecoder` 分帧。
    """

    data = bytes(frame)
    if len(data) != PROTOCOL_55AA_FRAME_SIZE:
        raise Protocol55AAError(f"frame must be {PROTOCOL_55AA_FRAME_SIZE} bytes, got {len(data)}")
    if data[:2] != PROTOCOL_55AA_HEADER:
        raise Protocol55AAError("invalid frame header; expected 55 AA")
    if data[2] != PROTOCOL_55AA_LENGTH:
        raise Protocol55AAError(f"invalid length byte; expected 0x{PROTOCOL_55AA_LENGTH:02X}")
    if data[-1] != PROTOCOL_55AA_TAIL:
        raise Protocol55AAError(f"invalid frame tail; expected 0x{PROTOCOL_55AA_TAIL:02X}")
    expected_checksum = xor_checksum(data[:-2])
    if data[-2] != expected_checksum:
        raise Protocol55AAError(f"checksum mismatch; received 0x{data[-2]:02X}, expected 0x{expected_checksum:02X}")
    payload = data[3 : 3 + PROTOCOL_55AA_PAYLOAD_SIZE]
    motors = tuple(
        RawMotorCommand(*struct.unpack_from(">HHH", payload, offset))
        for offset in range(0, PROTOCOL_55AA_PAYLOAD_SIZE, PROTOCOL_55AA_MOTOR_SIZE)
    )
    return Decoded55AAFrame(motors=motors, checksum=data[-2])


class Protocol55AAStreamDecoder:
    """从可能分包、粘包或夹杂坏字节的连续流中增量恢复有效帧。

    串口一次 read 不保证恰好返回53字节：一帧可能分多次到达，也可能一次收到多帧。
    ``feed`` 会保留未完成尾部；遇到错误长度、帧尾或 XOR 时只丢弃候选帧头的第一个
    字节，再继续寻找下一个 ``55 AA``。帧长固定，因此 payload 中出现 55 AA 无需转义。
    """

    def __init__(self) -> None:
        """创建空接收缓存；同一通讯连接应持续复用这个实例。"""

        self._buffer = bytearray()

    def feed(self, data: bytes | bytearray | memoryview) -> list[Decoded55AAFrame]:
        """追加本次收到的任意长度字节，并返回其中新完成的全部有效帧。"""

        self._buffer.extend(data)
        frames: list[Decoded55AAFrame] = []
        while True:
            header_index = self._buffer.find(PROTOCOL_55AA_HEADER)
            if header_index < 0:
                # 保留末尾单独的0x55：它可能是一次跨两次read到达的帧头第一个字节。
                trailing_header_byte = bool(self._buffer and self._buffer[-1] == PROTOCOL_55AA_HEADER[0])
                self._buffer[:] = self._buffer[-1:] if trailing_header_byte else b""
                break
            if header_index:
                del self._buffer[:header_index]
            if len(self._buffer) < 3:
                break
            if self._buffer[2] != PROTOCOL_55AA_LENGTH:
                del self._buffer[0]
                continue
            frame_size = 3 + self._buffer[2]
            if len(self._buffer) < frame_size:
                break
            candidate = bytes(self._buffer[:frame_size])
            try:
                frame = decode_raw_frame(candidate)
            except Protocol55AAError:
                del self._buffer[0]
                continue
            frames.append(frame)
            del self._buffer[:frame_size]
        return frames

    @property
    def buffered_bytes(self) -> int:
        """当前为下一次 ``feed`` 保留的未完成字节数。"""

        return len(self._buffer)


def _finite_number(value: Any, label: str) -> float:
    """把配置值转为有限浮点数；显式拒绝可被 Python 当作整数的 bool。"""

    if isinstance(value, bool):
        raise Protocol55AAError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Protocol55AAError(f"{label} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise Protocol55AAError(f"{label} must be finite, got {value!r}")
    return number


def _check_range(value: float, minimum: float, maximum: float, label: str) -> None:
    """检查物理量位于闭区间内；越界立即报错而不是裁剪。"""

    if not math.isfinite(value):
        raise Protocol55AAError(f"{label} must be finite")
    if value < minimum or value > maximum:
        raise Protocol55AAError(f"{label}={value} is outside [{minimum}, {maximum}]")


def _quantize(value: float, minimum: float, maximum: float, label: str) -> int:
    """把闭区间物理量线性映射到 uint16，并使用明确的四舍五入规则。"""

    _check_range(value, minimum, maximum, label)
    normalized = (value - minimum) / (maximum - minimum)
    # 显式采用 round-half-up，确保所有 Python 平台得到完全相同的量化结果。
    return min(UINT16_MAX, max(0, int(math.floor(normalized * UINT16_MAX + 0.5))))


@dataclass(frozen=True)
class MotorQuantization:
    """一个固定电机槽位的关节到电机标定、量化和软安全限制。

    编码前只做一次关节侧到电机侧换算：

    * ``q_motor = zero_offset + direction * gear_ratio * q_joint`` [rad]；
    * ``v_motor = direction * gear_ratio * v_joint`` [rad/s]，保留正负方向；
    * ``a_motor = gear_ratio * abs(a_joint)`` [rad/s^2]，线路只传幅值/上限。

    ``slot`` 决定48B数据区位置，``joint_name`` 决定从公共关节向量取哪个值。
    ``position_min_rad``/``position_max_rad`` 是经过实机审核的电机侧软限位，不会改变
    固件固定的 ±12.57 rad 编码量程。速度固定编码范围为 ±20 rad/s，加速度固定为
    0..65.535 rad/s^2；``command_max_*`` 只能进一步收紧它们。

    ``trajectory`` 模式发送轨迹自身的加速度幅值；``configured`` 模式发送审核过的固定
    加速度上限。两种模式的速度字段都始终是带符号轨迹目标速度，绝不是固定最大速度。
    """

    slot: int
    joint_name: str
    position_min_rad: float
    position_max_rad: float
    velocity_full_scale_rad_s: float = PROTOCOL_55AA_VELOCITY_MAX_RAD_S
    acceleration_full_scale_rad_s2: float = PROTOCOL_55AA_ACCELERATION_MAX_RAD_S2
    zero_offset_rad: float = 0.0
    direction: int = 1
    gear_ratio: float = 1.0
    limit_source: str = "trajectory"
    command_max_velocity_rad_s: float | None = None
    command_max_acceleration_rad_s2: float | None = None

    def __post_init__(self) -> None:
        """验证槽位、标定、固定固件量程和可选安全上限之间的一致性。"""

        if (
            isinstance(self.slot, bool)
            or not isinstance(self.slot, int)
            or not 1 <= self.slot <= PROTOCOL_55AA_MOTOR_COUNT
        ):
            raise Protocol55AAError(f"slot must be an integer in [1, {PROTOCOL_55AA_MOTOR_COUNT}], got {self.slot!r}")
        if not isinstance(self.joint_name, str) or not self.joint_name:
            raise Protocol55AAError("joint_name must be a non-empty string")
        for name in (
            "position_min_rad",
            "position_max_rad",
            "velocity_full_scale_rad_s",
            "acceleration_full_scale_rad_s2",
            "zero_offset_rad",
            "gear_ratio",
        ):
            object.__setattr__(self, name, _finite_number(getattr(self, name), f"motor {self.slot} {name}"))
        if self.position_max_rad <= self.position_min_rad:
            raise Protocol55AAError(f"motor {self.slot} position_max_rad must exceed position_min_rad")
        if (
            self.position_min_rad < PROTOCOL_55AA_POSITION_MIN_RAD
            or self.position_max_rad > PROTOCOL_55AA_POSITION_MAX_RAD
        ):
            raise Protocol55AAError(f"motor {self.slot} position safety limits must be within [-12.57, 12.57] rad")
        if self.velocity_full_scale_rad_s != PROTOCOL_55AA_VELOCITY_MAX_RAD_S:
            raise Protocol55AAError(f"motor {self.slot} firmware velocity full scale is fixed at 20 rad/s")
        if self.acceleration_full_scale_rad_s2 != PROTOCOL_55AA_ACCELERATION_MAX_RAD_S2:
            raise Protocol55AAError(f"motor {self.slot} firmware acceleration full scale is fixed at 65.535 rad/s^2")
        if self.gear_ratio <= 0.0:
            raise Protocol55AAError(f"motor {self.slot} gear_ratio must be positive")
        if isinstance(self.direction, bool) or not isinstance(self.direction, int) or self.direction not in (-1, 1):
            raise Protocol55AAError(f"motor {self.slot} direction must be integer +1 or -1")
        source = str(self.limit_source).strip().lower()
        object.__setattr__(self, "limit_source", source)
        if source not in {"configured", "trajectory"}:
            raise Protocol55AAError(f"motor {self.slot} limit_source must be configured or trajectory")
        configured_values = (self.command_max_velocity_rad_s, self.command_max_acceleration_rad_s2)
        if source == "configured" and any(value is None for value in configured_values):
            raise Protocol55AAError(
                f"motor {self.slot} configured limits require command_max_velocity_rad_s and "
                "command_max_acceleration_rad_s2"
            )
        for name, full_scale in (
            ("command_max_velocity_rad_s", self.velocity_full_scale_rad_s),
            ("command_max_acceleration_rad_s2", self.acceleration_full_scale_rad_s2),
        ):
            value = getattr(self, name)
            if value is not None:
                limit = _finite_number(value, f"motor {self.slot} {name}")
                _check_range(limit, 0.0, full_scale, f"motor {self.slot} {name}")
                object.__setattr__(self, name, limit)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, default_limit_source: str = "trajectory") -> MotorQuantization:
        """从配置 JSON 中的一项构造并验证一个电机槽位。"""

        try:
            return cls(
                slot=value["slot"],
                joint_name=value["joint_name"],
                position_min_rad=value["position_min_rad"],
                position_max_rad=value["position_max_rad"],
                velocity_full_scale_rad_s=value.get("velocity_full_scale_rad_s", PROTOCOL_55AA_VELOCITY_MAX_RAD_S),
                acceleration_full_scale_rad_s2=value.get(
                    "acceleration_full_scale_rad_s2", PROTOCOL_55AA_ACCELERATION_MAX_RAD_S2
                ),
                zero_offset_rad=value.get("zero_offset_rad", 0.0),
                direction=value.get("direction", 1),
                gear_ratio=value.get("gear_ratio", 1.0),
                limit_source=str(value.get("limit_source", default_limit_source)),
                command_max_velocity_rad_s=value.get("command_max_velocity_rad_s"),
                command_max_acceleration_rad_s2=value.get("command_max_acceleration_rad_s2"),
            )
        except KeyError as exc:
            raise Protocol55AAError(f"motor configuration is missing {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            raise Protocol55AAError(f"invalid motor configuration: {exc}") from exc

    def encode(self, angle_rad: float, velocity_rad_s: float, acceleration_rad_s2: float) -> RawMotorCommand:
        """标定并量化一个关节的角度、带符号速度和角加速度。

        输入依次为关节侧角度 [rad]、速度 [rad/s]、角加速度 [rad/s^2]。先应用零位、
        方向和减速比，再检查电机侧软限位并量化。量化完成后还会按固件公式反解一次，
        再检查反解值是否仍在更紧的软限位内；这是为了防止整数舍入把一个边界内输入
        推到边界外。任何失败都抛错，不静默截断。

        加速度在线路上只携带非负幅值/上限，所以负导数会先取绝对值；速度则保留符号。
        由于 65535 为奇数，中间码无法精确表示 0 rad 或 0 rad/s，软限位必须为量化和
        固件 float32 反解保留余量。
        """

        angle = _finite_number(angle_rad, f"{self.joint_name} angle_rad")
        velocity = _finite_number(velocity_rad_s, f"{self.joint_name} velocity_rad_s")
        acceleration = _finite_number(acceleration_rad_s2, f"{self.joint_name} acceleration_rad_s2")
        motor_position = self.zero_offset_rad + self.direction * self.gear_ratio * angle
        motor_speed = self.direction * self.gear_ratio * velocity
        motor_acceleration = self.gear_ratio * abs(acceleration)
        velocity_limit = (
            self.velocity_full_scale_rad_s
            if self.command_max_velocity_rad_s is None
            else self.command_max_velocity_rad_s
        )
        acceleration_limit = (
            self.acceleration_full_scale_rad_s2
            if self.command_max_acceleration_rad_s2 is None
            else self.command_max_acceleration_rad_s2
        )
        _check_range(motor_position, self.position_min_rad, self.position_max_rad, f"{self.joint_name} motor position")
        _check_range(motor_speed, -velocity_limit, velocity_limit, f"{self.joint_name} target velocity")
        _check_range(motor_acceleration, 0.0, acceleration_limit, f"{self.joint_name} acceleration magnitude")
        if self.limit_source == "configured":
            motor_acceleration = acceleration_limit
        command = RawMotorCommand(
            target_position=_quantize(
                motor_position,
                PROTOCOL_55AA_POSITION_MIN_RAD,
                PROTOCOL_55AA_POSITION_MAX_RAD,
                f"{self.joint_name} motor position",
            ),
            max_speed=_quantize(
                motor_speed,
                PROTOCOL_55AA_VELOCITY_MIN_RAD_S,
                PROTOCOL_55AA_VELOCITY_MAX_RAD_S,
                f"{self.joint_name} target velocity",
            ),
            max_acceleration=int(math.floor(motor_acceleration * 1000.0 + 0.5)),
        )
        # 四舍五入不能让接收端实际目标越过更严格的安全限位；速度零无法被精确表示。
        _check_range(
            self.decode_position_rad(command.target_position),
            self.position_min_rad,
            self.position_max_rad,
            f"{self.joint_name} quantized motor position",
        )
        _check_range(
            self.decode_velocity_rad_s(command.target_velocity),
            -velocity_limit,
            velocity_limit,
            f"{self.joint_name} quantized target velocity",
        )
        _check_range(
            self.decode_acceleration_rad_s2(command.acceleration_limit),
            0.0,
            acceleration_limit,
            f"{self.joint_name} quantized acceleration magnitude",
        )
        return command

    def decode_position_rad(self, raw_position: int) -> float:
        """按固件公式把位置码反解为电机侧角度 [rad]。"""

        RawMotorCommand(raw_position, 0, 0)
        return (raw_position / UINT16_MAX) * 25.14 - 12.57

    def decode_velocity_rad_s(self, raw_velocity: int) -> float:
        """按固件公式把偏置速度码反解为电机侧有符号速度 [rad/s]。"""

        RawMotorCommand(0, raw_velocity, 0)
        return (raw_velocity / UINT16_MAX) * 40.0 - 20.0

    def decode_acceleration_rad_s2(self, raw_acceleration: int) -> float:
        """按 ``raw/1000`` 反解无符号加速度幅值/上限 [rad/s^2]。"""

        RawMotorCommand(0, 0, raw_acceleration)
        return raw_acceleration / 1000.0


@dataclass(frozen=True)
class Protocol55AAConfig:
    """完整的8电机槽位、标定和安全量化配置。"""

    motors: tuple[MotorQuantization, ...]

    def __post_init__(self) -> None:
        """强制槽位恰为1..8、关节不重复且顺序符合线路协议。"""

        ordered = tuple(sorted(self.motors, key=lambda motor: motor.slot))
        if len(ordered) != PROTOCOL_55AA_MOTOR_COUNT:
            raise Protocol55AAError(f"configuration must contain {PROTOCOL_55AA_MOTOR_COUNT} motors")
        if tuple(motor.slot for motor in ordered) != tuple(range(1, PROTOCOL_55AA_MOTOR_COUNT + 1)):
            raise Protocol55AAError(f"motor slots must be exactly 1 through {PROTOCOL_55AA_MOTOR_COUNT}")
        names = tuple(motor.joint_name for motor in ordered)
        if len(set(names)) != len(names):
            raise Protocol55AAError("joint_name values must be unique")
        if names != PROTOCOL_55AA_JOINT_ORDER:
            raise Protocol55AAError(
                "motor slot order must be: "
                + ", ".join(f"{slot}:{name}" for slot, name in enumerate(PROTOCOL_55AA_JOINT_ORDER, start=1))
            )
        object.__setattr__(self, "motors", ordered)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Protocol55AAConfig:
        """从已解析 JSON 对象加载 v2 配置并执行完整一致性检查。

        v1 使用不兼容的小端/可变量程，因此明确拒绝。``example_only=true`` 的模板也会
        被拒绝，必须复制后填入实测软限位、零位、方向、减速比与指令上限，并由人员
        审核后才能改为 false。
        """

        schema = str(payload.get("schema", ""))
        if schema != PROTOCOL_55AA_CONFIG_SCHEMA:
            if schema == "exoskeleton-55aa-config/v1":
                raise Protocol55AAError(
                    "v1 uses incompatible little-endian/configurable scales; migrate to the v2 template "
                    "with big-endian fixed firmware ranges and regenerate binary files"
                )
            raise Protocol55AAError(f"configuration schema must be {PROTOCOL_55AA_CONFIG_SCHEMA!r}, got {schema!r}")
        if payload.get("example_only", False) is not False:
            raise Protocol55AAError(
                "example_only configuration is not deployable; copy it, replace every calibration value, "
                "and set example_only to false"
            )
        byte_order = str(payload.get("byte_order", PROTOCOL_55AA_BYTE_ORDER)).strip().lower()
        if byte_order != PROTOCOL_55AA_BYTE_ORDER:
            raise Protocol55AAError("protocol v2 fixes all uint16 fields to big-endian, matching the firmware")
        configured_length = payload.get("length", PROTOCOL_55AA_LENGTH)
        if isinstance(configured_length, str):
            try:
                configured_length = int(configured_length, 0)
            except ValueError as exc:
                raise Protocol55AAError(f"length must be 0x{PROTOCOL_55AA_LENGTH:02X}") from exc
        if configured_length != PROTOCOL_55AA_LENGTH:
            raise Protocol55AAError(f"protocol v2 length must be 0x{PROTOCOL_55AA_LENGTH:02X}")
        values = payload.get("motors")
        if not isinstance(values, list):
            raise Protocol55AAError("configuration must contain a motors array")
        default_limit_source = str(payload.get("limit_source", "trajectory"))
        motors = tuple(
            MotorQuantization.from_mapping(value, default_limit_source=default_limit_source)
            if isinstance(value, Mapping)
            else _raise_invalid_motor(index)
            for index, value in enumerate(values)
        )
        return cls(motors)

    @classmethod
    def from_file(cls, path: str | Path) -> Protocol55AAConfig:
        """从磁盘读取 JSON 协议配置，并交给 :meth:`from_mapping` 严格验证。"""

        config_path = Path(path).expanduser()
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Protocol55AAError(f"cannot read protocol configuration {config_path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise Protocol55AAError("protocol configuration must be a JSON object")
        return cls.from_mapping(payload)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """返回按电机槽位1到8排列的源关节名称。"""

        return tuple(motor.joint_name for motor in self.motors)

    def encode(
        self,
        joint_names: Sequence[str],
        angle_rad: Sequence[float],
        velocity_rad_s: Sequence[float],
        acceleration_rad_s2: Sequence[float],
    ) -> bytes:
        """按关节名称选值、重排到槽位1..8、逐电机标定，最后返回一帧53B数据。

        输入的 ``joint_names`` 可以采用策略公共顺序，不要求已经按电机号排列；角度、
        速度、加速度三个向量必须与它等长且使用 rad、rad/s、rad/s^2。按名称映射而非
        直接按下标复制，是为了正确处理“策略右腿优先、协议左右交错”的顺序差异。
        """

        names = tuple(str(name) for name in joint_names)
        if len(set(names)) != len(names):
            raise Protocol55AAError("source joint_names contains duplicates")
        values = tuple(tuple(sequence) for sequence in (angle_rad, velocity_rad_s, acceleration_rad_s2))
        if any(len(sequence) != len(names) for sequence in values):
            raise Protocol55AAError("joint_names and target vectors must have matching lengths")
        index_by_name = {name: index for index, name in enumerate(names)}
        missing = [name for name in self.joint_names if name not in index_by_name]
        if missing:
            raise Protocol55AAError(f"source target is missing protocol joints: {missing}")
        commands = []
        for motor in self.motors:
            index = index_by_name[motor.joint_name]
            commands.append(motor.encode(values[0][index], values[1][index], values[2][index]))
        return encode_raw_frame(commands)


def _raise_invalid_motor(index: int) -> MotorQuantization:
    """把电机数组中“不是 JSON 对象”的位置转换为带下标的明确异常。"""

    raise Protocol55AAError(f"motors[{index}] must be a JSON object")


__all__ = [
    "Decoded55AAFrame",
    "MotorQuantization",
    "PROTOCOL_55AA_ACCELERATION_MAX_RAD_S2",
    "PROTOCOL_55AA_ACCELERATION_RESOLUTION_RAD_S2",
    "PROTOCOL_55AA_BYTE_ORDER",
    "PROTOCOL_55AA_CONFIG_SCHEMA",
    "PROTOCOL_55AA_FRAME_SIZE",
    "PROTOCOL_55AA_HEADER",
    "PROTOCOL_55AA_JOINT_ORDER",
    "PROTOCOL_55AA_LENGTH",
    "PROTOCOL_55AA_MOTOR_COUNT",
    "PROTOCOL_55AA_MOTOR_SIZE",
    "PROTOCOL_55AA_PAYLOAD_SIZE",
    "PROTOCOL_55AA_POSITION_MAX_RAD",
    "PROTOCOL_55AA_POSITION_MIN_RAD",
    "PROTOCOL_55AA_TAIL",
    "PROTOCOL_55AA_VELOCITY_MAX_RAD_S",
    "PROTOCOL_55AA_VELOCITY_MIN_RAD_S",
    "Protocol55AAConfig",
    "Protocol55AAError",
    "Protocol55AAStreamDecoder",
    "RawMotorCommand",
    "decode_raw_frame",
    "encode_raw_frame",
    "xor_checksum",
]
