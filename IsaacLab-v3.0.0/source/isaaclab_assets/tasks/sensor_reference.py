# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-device reference trajectories for optional sensor-imitation rewards.

The loader intentionally accepts a small, explicit schema instead of trying to
parse a ROS bag inside the simulator.  Convert a recording offline, validate
its units and joint order, and pass an ``.npz`` or ``.pt`` file to the command
configuration.  The resulting tensors are sampled without filesystem access
while an episode is running.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .robot_contract import ACTUATED_JOINT_NAMES, CONTROL_DT_S


class SensorReferenceFormatError(ValueError):
    """Raised when a reference recording does not satisfy the expected schema."""


@dataclass(frozen=True)
class SensorReference:
    """A time-aligned sensor trajectory stored as tensors.

    All tensors use the canonical right-leg-then-left-leg order.  Time is in
    seconds, joint angles in radians, joint rates and gyros in rad/s, and
    contact values are normalized to ``[0, 1]``.
    """

    t_s: torch.Tensor
    joint_pos_rad: torch.Tensor
    joint_vel_rad_s: torch.Tensor | None = None
    gyro_rad_s: torch.Tensor | None = None
    contact_lr: torch.Tensor | None = None
    gait_id: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return int(self.joint_pos_rad.shape[0])

    def sample(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Gather samples for environment indices, wrapping at the trajectory end."""

        if indices.ndim != 1:
            raise ValueError(f"reference indices must be one-dimensional, got {tuple(indices.shape)}")
        safe_indices = indices.remainder(self.length)
        result = {
            "joint_pos_rad": self.joint_pos_rad[safe_indices],
        }
        for name in ("joint_vel_rad_s", "gyro_rad_s", "contact_lr", "gait_id"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value[safe_indices]
        return result


class SensorReferenceBuffer:
    """A trajectory plus per-environment playback cursors."""

    def __init__(self, reference: SensorReference, device: torch.device, loop: bool = True):
        if reference.length < 2:
            raise SensorReferenceFormatError("reference trajectory must contain at least two samples")
        self.reference = _to_device(reference, device)
        self.device = device
        self.loop = bool(loop)
        self.indices = torch.zeros(0, dtype=torch.long, device=device)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        device: torch.device,
        target_dt_s: float = CONTROL_DT_S,
        loop: bool = True,
    ) -> SensorReferenceBuffer:
        """Load and resample one recording for the policy control period."""

        reference = load_sensor_reference(path, target_dt_s=target_dt_s)
        return cls(reference, device=device, loop=loop)

    @property
    def length(self) -> int:
        return self.reference.length

    def reset(self, env_ids: torch.Tensor | slice, num_envs: int, random_start: bool = False) -> None:
        """Reset selected cursors, optionally choosing random trajectory starts."""

        if self.indices.numel() == 0 or self.indices.shape[0] != num_envs:
            self.indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, slice):
            ids = torch.arange(num_envs, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() > 0:
            if random_start:
                self.indices[ids] = torch.randint(self.length, (ids.numel(),), device=self.device)
            else:
                self.indices[ids] = 0

    def advance(self, steps: int = 1) -> None:
        """Advance all cursors by a number of policy samples."""

        if steps < 0:
            raise ValueError("reference playback steps must be non-negative")
        if self.indices.numel() == 0 or steps == 0:
            return
        if self.loop:
            self.indices.add_(steps).remainder_(self.length)
        else:
            self.indices.add_(steps).clamp_max_(self.length - 1)

    def sample(self) -> dict[str, torch.Tensor] | None:
        """Return the current per-environment reference sample."""

        if self.indices.numel() == 0:
            return None
        return self.reference.sample(self.indices)


def load_sensor_reference(path: str | Path, target_dt_s: float = CONTROL_DT_S) -> SensorReference:
    """Load, calibrate, validate, and resample a sensor recording.

    Supported files are ``.npz``, ``.pt``/``.pth``, ``.json`` and CSV.  The
    required field is ``joint_pos_rad`` with shape ``[T, 8]``.  Optional fields
    are ``t_s``, ``joint_vel_rad_s`` ``[T, 8]``, ``gyro_rad_s`` ``[T, 6, 3]``,
    ``contact_lr`` ``[T, 2]`` and ``gait_id`` ``[T]``.  A recording may include
    ``joint_zero_rad`` and ``joint_sign`` metadata for encoder calibration.
    """

    if target_dt_s <= 0.0:
        raise ValueError("target_dt_s must be positive")
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"sensor reference file does not exist: {path}")
    payload = _read_payload(path)
    reference = _reference_from_mapping(payload, target_dt_s=target_dt_s)
    return _resample_reference(reference, target_dt_s)


def _read_payload(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - Isaac Lab normally supplies numpy
            raise SensorReferenceFormatError(".npz reference files require numpy") from exc
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if suffix in {".pt", ".pth"}:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch versions before the weights_only argument
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise SensorReferenceFormatError(".pt reference files must contain a mapping")
        return payload
    if suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping):
            raise SensorReferenceFormatError(".json reference files must contain an object")
        return payload
    if suffix == ".csv":
        return _read_csv(path)
    raise SensorReferenceFormatError(f"unsupported reference extension '{path.suffix}'; use .npz, .pt, .json, or .csv")


def _read_csv(path: Path) -> Mapping[str, Any]:
    """Read the compact column form used by the data-preparation guide."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SensorReferenceFormatError("CSV reference file has no data rows")
    columns = list(rows[0])

    def column(*names: str) -> list[float] | None:
        for name in names:
            if name in columns:
                return [float(row[name]) for row in rows]
        return None

    result: dict[str, Any] = {"t_s": column("t_s", "time_s", "time")}
    q_columns = _numbered_columns(columns, ("joint_pos_rad", "q_rad", "q"), len(ACTUATED_JOINT_NAMES))
    if q_columns is None:
        raise SensorReferenceFormatError("CSV must provide joint_pos_rad[0..7] (or q_rad[0..7]) columns")
    result["joint_pos_rad"] = [[float(row[name]) for name in q_columns] for row in rows]
    for field, prefixes, count in (
        ("joint_vel_rad_s", ("joint_vel_rad_s", "qdot_rad_s", "qdot"), 8),
        ("contact_lr", ("contact_lr", "contact"), 2),
    ):
        names = _numbered_columns(columns, prefixes, count)
        if names is not None:
            result[field] = [[float(row[name]) for name in names] for row in rows]
    gyro_names = _numbered_columns(columns, ("gyro_rad_s", "gyro"), 18)
    if gyro_names is not None:
        result["gyro_rad_s"] = [
            [[float(row[name]) for name in gyro_names[i : i + 3]] for i in range(0, 18, 3)] for row in rows
        ]
    return {key: value for key, value in result.items() if value is not None}


def _numbered_columns(columns: list[str], prefixes: tuple[str, ...], count: int) -> list[str] | None:
    for prefix in prefixes:
        candidates = []
        for index in range(count):
            names = (f"{prefix}[{index}]", f"{prefix}_{index}", f"{prefix}{index}")
            match = next((name for name in names if name in columns), None)
            if match is None:
                break
            candidates.append(match)
        if len(candidates) == count:
            return candidates
    return None


def _reference_from_mapping(payload: Mapping[str, Any], target_dt_s: float) -> SensorReference:
    def get(*names: str) -> Any:
        for name in names:
            if name in payload:
                return payload[name]
        return None

    joint_pos = _tensor(get("joint_pos_rad", "joint_pos", "q_rad", "q"), "joint_pos_rad")
    if joint_pos.ndim != 2:
        raise SensorReferenceFormatError(f"joint_pos_rad must have shape [T, 8], got {tuple(joint_pos.shape)}")
    if joint_pos.shape[-1] != len(ACTUATED_JOINT_NAMES) and joint_pos.shape[0] == len(ACTUATED_JOINT_NAMES):
        joint_pos = joint_pos.transpose(0, 1)
    if joint_pos.shape[-1] != len(ACTUATED_JOINT_NAMES):
        raise SensorReferenceFormatError(f"joint_pos_rad must have shape [T, 8], got {tuple(joint_pos.shape)}")

    zero = get("joint_zero_rad", "zero_rad")
    sign = get("joint_sign", "sign")
    if zero is not None:
        zero_tensor = _tensor(zero, "joint_zero_rad").reshape(-1)
        if zero_tensor.numel() != joint_pos.shape[-1]:
            raise SensorReferenceFormatError("joint_zero_rad must contain eight values")
        joint_pos = joint_pos - zero_tensor
    if sign is not None:
        sign_tensor = _tensor(sign, "joint_sign").reshape(-1)
        if sign_tensor.numel() != joint_pos.shape[-1] or not torch.all(torch.abs(sign_tensor) == 1):
            raise SensorReferenceFormatError("joint_sign must contain eight values, each +1 or -1")
        joint_pos = joint_pos * sign_tensor

    length = joint_pos.shape[0]
    t_value = get("t_s", "time_s", "time", "timestamps_s")
    if t_value is None:
        t_s = torch.arange(length, dtype=torch.float32) * target_dt_s
    else:
        t_s = _tensor(t_value, "t_s").reshape(-1)
        if t_s.numel() != length:
            raise SensorReferenceFormatError("t_s length must match joint_pos_rad")
        t_s = t_s - t_s[0]
    if length < 2 or not torch.isfinite(joint_pos).all() or not torch.isfinite(t_s).all():
        raise SensorReferenceFormatError("reference time and joint positions must be finite and contain >=2 samples")
    if torch.any(torch.diff(t_s) <= 0):
        raise SensorReferenceFormatError("t_s must be strictly increasing")

    joint_vel = _optional_tensor(get("joint_vel_rad_s", "joint_vel", "qdot_rad_s", "qdot"), length, (8,))
    gyro = _optional_tensor(get("gyro_rad_s", "gyro", "imu_gyro_rad_s"), length, (6, 3))
    contact = _optional_tensor(get("contact_lr", "contact", "foot_contact"), length, (2,))
    gait = _optional_tensor(get("gait_id", "gait"), length, ())
    if contact is not None:
        # Both binary contact and force-like recordings are accepted.  Values
        # above one are interpreted as newtons and normalized by 5 N.
        contact = torch.where(contact.abs() > 1.0, contact.abs() / 5.0, contact).clamp(0.0, 1.0)
    if gait is not None:
        gait = gait.round().to(dtype=torch.long).reshape(length)
    return SensorReference(t_s, joint_pos, joint_vel, gyro, contact, gait)


def _tensor(value: Any, name: str) -> torch.Tensor:
    if value is None:
        raise SensorReferenceFormatError(f"reference is missing required field '{name}'")
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise SensorReferenceFormatError(f"reference field '{name}' is not numeric") from exc
    return tensor


def _optional_tensor(value: Any, length: int, trailing_shape: tuple[int, ...]) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = _tensor(value, "optional")
    expected = (length, *trailing_shape)
    if tensor.shape != expected:
        # Permit flattened [T, 18] gyro input.
        if trailing_shape == (6, 3) and tensor.shape == (length, 18):
            tensor = tensor.reshape(length, 6, 3)
        else:
            raise SensorReferenceFormatError(f"optional field must have shape {expected}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise SensorReferenceFormatError("optional reference fields must be finite")
    return tensor


def _resample_reference(reference: SensorReference, target_dt_s: float) -> SensorReference:
    """Linearly resample continuous channels and nearest-neighbor labels."""

    old_t = reference.t_s
    duration = float(old_t[-1])
    count = max(2, int(torch.floor(torch.tensor(duration / target_dt_s)).item()) + 1)
    new_t = torch.arange(count, dtype=torch.float32) * target_dt_s
    new_t[-1] = min(float(new_t[-1]), duration)

    def interpolate(values: torch.Tensor) -> torch.Tensor:
        right = torch.searchsorted(old_t, new_t).clamp(1, old_t.numel() - 1)
        left = right - 1
        alpha = ((new_t - old_t[left]) / (old_t[right] - old_t[left])).reshape(-1, *([1] * (values.ndim - 1)))
        return values[left] * (1.0 - alpha) + values[right] * alpha

    def nearest(values: torch.Tensor) -> torch.Tensor:
        right = torch.searchsorted(old_t, new_t).clamp(1, old_t.numel() - 1)
        left = right - 1
        choose_right = (new_t - old_t[left]) > (old_t[right] - new_t)
        return values[torch.where(choose_right, right, left)]

    return SensorReference(
        new_t,
        interpolate(reference.joint_pos_rad),
        None if reference.joint_vel_rad_s is None else interpolate(reference.joint_vel_rad_s),
        None if reference.gyro_rad_s is None else interpolate(reference.gyro_rad_s),
        None if reference.contact_lr is None else interpolate(reference.contact_lr).clamp(0.0, 1.0),
        None if reference.gait_id is None else nearest(reference.gait_id),
    )


def _to_device(reference: SensorReference, device: torch.device) -> SensorReference:
    return SensorReference(
        *(value.to(device=device) if value is not None else None for value in (
            reference.t_s,
            reference.joint_pos_rad,
            reference.joint_vel_rad_s,
            reference.gyro_rad_s,
            reference.contact_lr,
            reference.gait_id,
        ))
    )


__all__ = [
    "SensorReference",
    "SensorReferenceBuffer",
    "SensorReferenceFormatError",
    "load_sensor_reference",
]
