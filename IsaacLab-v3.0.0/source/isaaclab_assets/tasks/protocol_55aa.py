# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Encode and validate the eight-motor ``55 AA`` command protocol.

The wire format is deliberately independent of Isaac Sim and Torch so the
same implementation can be reused by a playback recorder, an industrial-PC
process, and unit tests. Physical values are converted to unsigned 16-bit
fields only after an explicit per-motor calibration has been loaded. Version 2
matches the firmware's big-endian, biased signed position/velocity fields and
unsigned acceleration magnitude. The frame itself carries no version marker.
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
"""Fixed motor slots: left/right ankle, knee, hip, then steering joint."""


class Protocol55AAError(ValueError):
    """Raised when a protocol frame or quantization configuration is invalid."""


def xor_checksum(data: bytes | bytearray | memoryview) -> int:
    """Return the byte-wise XOR of *data*."""

    checksum = 0
    for value in data:
        checksum ^= int(value)
    return checksum


@dataclass(frozen=True)
class RawMotorCommand:
    """One motor's unsigned wire codes, not physical values.

    ``max_speed`` is a legacy field name for the biased signed target velocity
    in v2; use ``target_velocity`` for new readers. ``max_acceleration`` remains
    an unsigned acceleration magnitude/limit, never a signed derivative.
    """

    target_position: int
    max_speed: int
    max_acceleration: int

    def __post_init__(self) -> None:
        for name in ("target_position", "max_speed", "max_acceleration"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT16_MAX:
                raise Protocol55AAError(f"{name} must be an integer in [0, 65535], got {value!r}")

    @property
    def target_velocity(self) -> int:
        """Biased signed velocity code (legacy ``max_speed`` alias)."""

        return self.max_speed

    @property
    def acceleration_limit(self) -> int:
        """Unsigned acceleration code (legacy ``max_acceleration`` alias)."""

        return self.max_acceleration


@dataclass(frozen=True)
class Decoded55AAFrame:
    """Validated raw command frame."""

    motors: tuple[RawMotorCommand, ...]
    checksum: int

    def __post_init__(self) -> None:
        if len(self.motors) != PROTOCOL_55AA_MOTOR_COUNT:
            raise Protocol55AAError(f"a frame must contain {PROTOCOL_55AA_MOTOR_COUNT} motor records")


def encode_raw_frame(motors: Sequence[RawMotorCommand]) -> bytes:
    """Encode eight raw motor records into one 53-byte command frame.

    All 16-bit fields are big-endian, matching ``(high << 8) | low``. The checksum covers the header,
    length byte, and 48-byte payload; it excludes the checksum byte and tail.
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
    """Validate and decode one complete 53-byte command frame."""

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
    """Incrementally recover valid frames from a byte stream.

    Bad length, tail, or checksum values discard one candidate header byte and
    resume scanning. Payload bytes need no escaping because the fixed length
    determines the frame boundary.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes | bytearray | memoryview) -> list[Decoded55AAFrame]:
        """Append stream bytes and return every newly completed valid frame."""

        self._buffer.extend(data)
        frames: list[Decoded55AAFrame] = []
        while True:
            header_index = self._buffer.find(PROTOCOL_55AA_HEADER)
            if header_index < 0:
                # Retain a trailing 0x55 because it may be the first byte of a
                # header split across two reads.
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
        """Number of incomplete bytes retained for the next call."""

        return len(self._buffer)


def _finite_number(value: Any, label: str) -> float:
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
    if not math.isfinite(value):
        raise Protocol55AAError(f"{label} must be finite")
    if value < minimum or value > maximum:
        raise Protocol55AAError(f"{label}={value} is outside [{minimum}, {maximum}]")


def _quantize(value: float, minimum: float, maximum: float, label: str) -> int:
    _check_range(value, minimum, maximum, label)
    normalized = (value - minimum) / (maximum - minimum)
    # Round-half-up is explicit and is identical on every Python platform.
    return min(UINT16_MAX, max(0, int(math.floor(normalized * UINT16_MAX + 0.5))))


@dataclass(frozen=True)
class MotorQuantization:
    """Calibration and motor-side safety limits for one fixed slot.

    Position is first converted from the simulation joint to motor-side angle:
    ``q_motor = zero_offset + direction * gear_ratio * q_joint`` [rad]. The
    velocity target is ``direction * gear_ratio * v_joint`` [rad/s], preserving
    its sign. Acceleration is an unsigned magnitude [rad/s^2].

    The firmware quantization ranges are fixed; ``position_min_rad`` and
    ``position_max_rad`` are tighter safety limits, not configurable scales.
    ``command_max_velocity_rad_s`` and ``command_max_acceleration_rad_s2`` are
    optional, tighter magnitude limits in ``trajectory`` mode. ``configured``
    mode requires both limits and sends the fixed acceleration limit, while
    still sending the signed trajectory velocity (never a fixed speed).
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
        """Build one motor configuration from a JSON object."""

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
        """Quantize joint angle [rad], signed speed [rad/s], and acceleration [rad/s^2].

        All calibrated values are checked before rounding. The acceleration
        field carries only the magnitude/limit; a negative derivative is not
        a negative wire command. No out-of-range input is silently clipped.
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
        # Rounding must not push the actual receiver target past a tighter
        # safety limit. A zero velocity cannot be represented exactly.
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
        """Convert one raw target-position value back to motor-side radians."""

        RawMotorCommand(raw_position, 0, 0)
        return (raw_position / UINT16_MAX) * 25.14 - 12.57

    def decode_velocity_rad_s(self, raw_velocity: int) -> float:
        """Decode the motor-side signed velocity [rad/s], matching the firmware."""

        RawMotorCommand(0, raw_velocity, 0)
        return (raw_velocity / UINT16_MAX) * 40.0 - 20.0

    def decode_acceleration_rad_s2(self, raw_acceleration: int) -> float:
        """Decode the unsigned acceleration magnitude/limit [rad/s^2]."""

        RawMotorCommand(0, 0, raw_acceleration)
        return raw_acceleration / 1000.0


@dataclass(frozen=True)
class Protocol55AAConfig:
    """Complete fixed-order eight-motor quantization configuration."""

    motors: tuple[MotorQuantization, ...]

    def __post_init__(self) -> None:
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
        """Load and validate a configuration mapping."""

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
        """Load a JSON protocol configuration file."""

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
        """Source joint names in fixed motor-slot order."""

        return tuple(motor.joint_name for motor in self.motors)

    def encode(
        self,
        joint_names: Sequence[str],
        angle_rad: Sequence[float],
        velocity_rad_s: Sequence[float],
        acceleration_rad_s2: Sequence[float],
    ) -> bytes:
        """Select eight named joints, quantize them, and return one wire frame."""

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
