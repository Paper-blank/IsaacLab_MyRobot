# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset and mapping utilities for vision/IMU motion imitation.

The recorder used by this project stores one trial below::

    records/<subject>/<date>/<task-id>/TrialXX/

Each trial normally contains ``*_imu_01.csv`` ... ``*_imu_11.csv``, two RGB
videos and two DVXplorer ``.aedat4`` recordings.  The loader deliberately
keeps media decoding optional: IMU data is parsed eagerly and video/event
streams are decoded only when a sampled window asks for them.  In particular,
an event preview video is never treated as an event stream.

The module has no Isaac Sim imports and can therefore be used by data
preparation scripts and unit tests without starting Kit.

Recorded direction conventions are normalized at load time.  The default
``sensor_coordinate_mapping.yaml`` flips the four sagittal hip/knee channels
and maps the IMU axes into the articulation frame; pass ``calibration_path``
to :class:`ExoskeletonImitationDataset` for a site-specific sign map.
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import math
import random
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

_LOGGER = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TASK_MAPPING = _REPO_ROOT / "task_mapping.yaml"
_DEFAULT_IMU_MAPPING = _REPO_ROOT / "imu_mapping.yaml"
_DEFAULT_COORDINATE_MAPPING = _REPO_ROOT / "sensor_coordinate_mapping.yaml"
_PACKAGE_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"

TASK_IDS = (
    "BT01",
    "BT02",
    "BT03",
    "BT04",
    "BT05",
    "AT01",
    "AT02",
    "AT03",
    "AT04",
    "AT05",
)
IMU_IDS = tuple(f"imu_{index:02d}" for index in range(1, 7))


class DatasetFormatError(ValueError):
    """Raised when a recording cannot be interpreted safely."""


# The recorder and the URDF use the same right-leg-then-left-leg order.  The
# hip/knee axes are mirrored across the sagittal plane, so the data-to-
# articulation signs are side-specific: right hip ``+``, right knee ``-``,
# left hip ``-``, left knee ``+``.  Turn and ankle signs remain configurable
# because their encoder conventions cannot be inferred from the joint axis
# alone.  A site-specific YAML can override this tuple after a single-joint
# calibration trial.
DEFAULT_JOINT_SIGN = (1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0)
# Legacy fallback signs for callers that do not identify a recorder sensor.
# The six-channel dataset path uses the explicit left/right matrices below and
# therefore does not apply these vectors a second time.
DEFAULT_IMU_ACCEL_SIGN = (-1.0, -1.0, 1.0)
DEFAULT_IMU_GYRO_SIGN = (-1.0, -1.0, 1.0)
DEFAULT_IMU_ORIENTATION_SIGN = (1.0, -1.0, 1.0)
# The recorder's retained channels alternate between legs: IMU-01/03/05 are
# mounted on the left leg and IMU-02/04/06 on the right leg.  Within each
# side, 01/02 are ankles, 03/04 are knees, and 05/06 are hips.  IMU-07..11
# are intentionally ignored by the imitation loader.  A matrix row is one
# simulation axis expressed in dataset coordinates.  For example, the
# right-leg matrix maps ``(dataset_x, dataset_y, dataset_z)`` to
# ``(joint_y, joint_x, -joint_z)``.
DEFAULT_IMU_SENSOR_SIDES = ("left", "right", "left", "right", "left", "right")
DEFAULT_IMU_AXIS_TRANSFORMS = {
    "right": (
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    ),
    "left": (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    ),
}

_IDENTITY_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
DEFAULT_IMU_JOINT_LINK_ROTATIONS = {
    f"imu_{index:02d}": _IDENTITY_ROTATION for index in range(1, 7)
}
DEFAULT_IMU_LINK_SENSOR_ROTATIONS = {
    f"imu_{index:02d}": _IDENTITY_ROTATION for index in range(1, 7)
}


@dataclass(frozen=True)
class SensorCoordinateCalibration:
    """Map recorded channels into the simulation convention.

    Leg sensors use the explicit signed-permutation matrices above, which
    handle the mirrored left/right mounting frames without changing the
    observation shape or the PPO contract.  The legacy base signs remain
    available for callers that explicitly select a ``base`` sensor side (or
    omit ``sensor_index``), but no retained dataset channel uses that side by
    default.
    """

    joint_sign: tuple[float, ...] = DEFAULT_JOINT_SIGN
    imu_orientation_sign: tuple[float, ...] = DEFAULT_IMU_ORIENTATION_SIGN
    imu_gyro_sign: tuple[float, ...] = DEFAULT_IMU_GYRO_SIGN
    imu_accel_sign: tuple[float, ...] = DEFAULT_IMU_ACCEL_SIGN
    imu_sensor_sides: tuple[str, ...] = DEFAULT_IMU_SENSOR_SIDES
    imu_axis_transforms: Mapping[str, tuple[tuple[float, ...], ...]] | None = field(
        default_factory=lambda: dict(DEFAULT_IMU_AXIS_TRANSFORMS)
    )
    # Physical mounting rotations are intentionally separate from the legacy
    # signed axis map.  The latter contains the project's left/right data
    # convention (the left matrix is a reflection), while these two matrices
    # are real frame rotations and must remain right-handed.
    imu_joint_link_rotations: Mapping[str, tuple[tuple[float, ...], ...]] = field(
        default_factory=lambda: dict(DEFAULT_IMU_JOINT_LINK_ROTATIONS)
    )
    imu_link_sensor_rotations: Mapping[str, tuple[tuple[float, ...], ...]] = field(
        default_factory=lambda: dict(DEFAULT_IMU_LINK_SENSOR_ROTATIONS)
    )
    imu_orientation_mode: str = "euler_components"

    def _sensor_side(self, sensor_index: int | None, sensor_side: str | None) -> str | None:
        """Resolve a recorder index (01/02 ankle, 03/04 knee, 05/06 hip) to a side."""

        if sensor_side is not None:
            return _normalise_sensor_side(sensor_side)
        if sensor_index is None:
            return None
        if sensor_index < 1 or sensor_index > len(self.imu_sensor_sides):
            raise ValueError(
                f"sensor_index must be in [1, {len(self.imu_sensor_sides)}], got {sensor_index}"
            )
        return self.imu_sensor_sides[sensor_index - 1]

    def _apply_axis_transform(self, values: np.ndarray, sensor_side: str | None) -> np.ndarray | None:
        """Apply a configured leg matrix, or return ``None`` for legacy signs."""

        if sensor_side in (None, "base") or self.imu_axis_transforms is None:
            return None
        matrix = self.imu_axis_transforms.get(sensor_side)
        if matrix is None:
            return None
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[-1] != 3:
            raise ValueError(f"IMU values must have a final dimension of 3, got {array.shape}")
        return np.einsum("ij,...j->...i", np.asarray(matrix, dtype=array.dtype), array)

    def _sensor_key(self, sensor_index: int | None, sensor_side: str | None) -> str | None:
        if sensor_index is not None:
            return f"imu_{sensor_index:02d}"
        if sensor_side in ("left", "right"):
            return str(sensor_side)
        return None

    def imu_transform_matrix(
        self,
        *,
        sensor_index: int | None = None,
        sensor_side: str | None = None,
    ) -> np.ndarray | None:
        """Return ``R_joint_link @ R_link_sensor @ R_sensor_IMU``.

        ``R_sensor_IMU`` is the existing recorder-to-leg signed axis map.  It
        is kept as a general orthogonal map because the prescribed left-leg
        convention has determinant ``-1``; it is therefore suitable for
        vector/channel remapping but is not itself a physical quaternion
        rotation.  The two mounting matrices are physical rotations and are
        identity until measured from the USD/fixture or a calibration trial.
        """

        side = self._sensor_side(sensor_index, sensor_side)
        if side in (None, "base") or self.imu_axis_transforms is None:
            return None
        axis = self.imu_axis_transforms.get(side)
        if axis is None:
            return None
        key = self._sensor_key(sensor_index, side)
        joint_link = self.imu_joint_link_rotations.get(key, _IDENTITY_ROTATION) if key else _IDENTITY_ROTATION
        link_sensor = self.imu_link_sensor_rotations.get(key, _IDENTITY_ROTATION) if key else _IDENTITY_ROTATION
        return (
            np.asarray(joint_link, dtype=np.float64)
            @ np.asarray(link_sensor, dtype=np.float64)
            @ np.asarray(axis, dtype=np.float64)
        )

    @staticmethod
    def _euler_xyz_to_matrix(values: np.ndarray) -> np.ndarray:
        """Convert ``[roll, pitch, yaw]`` to an XYZ rotation matrix."""

        roll, pitch, yaw = np.moveaxis(np.asarray(values, dtype=np.float64), -1, 0)
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        sy, cy = np.sin(yaw), np.cos(yaw)
        matrix = np.empty((*np.asarray(values).shape[:-1], 3, 3), dtype=np.float64)
        matrix[..., 0, 0] = cy * cp
        matrix[..., 0, 1] = cy * sp * sr - sy * cr
        matrix[..., 0, 2] = cy * sp * cr + sy * sr
        matrix[..., 1, 0] = sy * cp
        matrix[..., 1, 1] = sy * sp * sr + cy * cr
        matrix[..., 1, 2] = sy * sp * cr - cy * sr
        matrix[..., 2, 0] = -sp
        matrix[..., 2, 1] = cp * sr
        matrix[..., 2, 2] = cp * cr
        return matrix

    @staticmethod
    def _matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
        """Convert XYZ rotation matrices to ``[roll, pitch, yaw]``."""

        matrix = np.asarray(matrix, dtype=np.float64)
        pitch = np.arcsin(np.clip(-matrix[..., 2, 0], -1.0, 1.0))
        cp = np.cos(pitch)
        roll = np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2])
        yaw = np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
        singular = np.abs(cp) < 1.0e-7
        if np.any(singular):
            # A deterministic gimbal-lock convention: preserve yaw=0 and
            # place the residual rotation in roll.
            roll = np.where(singular, np.arctan2(-matrix[..., 0, 1], matrix[..., 1, 1]), roll)
            yaw = np.where(singular, 0.0, yaw)
        return np.stack((roll, pitch, yaw), axis=-1)

    def _transform_orientation(self, orientation: np.ndarray, transform: np.ndarray) -> np.ndarray:
        values = np.asarray(orientation)
        if self.imu_orientation_mode == "euler_components":
            # Compatibility mode for the recorder's published Euler channels.
            # This is the established component convention, not quaternion
            # multiplication; use rotation_matrix after mounting calibration.
            return np.einsum("ij,...j->...i", transform.astype(values.dtype), values)
        if not np.isclose(np.linalg.det(transform), 1.0, atol=1.0e-5):
            raise DatasetFormatError(
                "orientation_mode=rotation_matrix requires a right-handed effective IMU transform; "
                "the configured left-leg axis convention is a reflection"
            )
        return self._matrix_to_euler_xyz(transform @ self._euler_xyz_to_matrix(values)).astype(values.dtype)

    def apply_imu(
        self,
        orientation: np.ndarray,
        gyro: np.ndarray,
        accel: np.ndarray,
        *,
        sensor_index: int | None = None,
        sensor_side: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return calibrated ``(orientation, gyro, accel)`` arrays.

        ``sensor_index`` is one-based and matches the recorder filename
        suffix (``imu_01`` through ``imu_06``).  For a retained leg sensor, the
        same coordinate transform is applied to all three vectors, exactly
        matching the requested dataset-to-joint mapping.  Omitting the index
        preserves the pre-existing global-sign behavior for external callers.
        """

        side = self._sensor_side(sensor_index, sensor_side)
        transform = self.imu_transform_matrix(sensor_index=sensor_index, sensor_side=sensor_side)
        if transform is not None:
            transformed = (
                self._transform_orientation(orientation, transform),
                np.einsum("ij,...j->...i", transform.astype(np.asarray(gyro).dtype), np.asarray(gyro)),
                np.einsum("ij,...j->...i", transform.astype(np.asarray(accel).dtype), np.asarray(accel)),
            )
        else:
            transformed = tuple(self._apply_axis_transform(value, side) for value in (orientation, gyro, accel))
        if all(value is not None for value in transformed):
            return transformed  # type: ignore[return-value]
        return (
            np.asarray(orientation) * np.asarray(self.imu_orientation_sign),
            np.asarray(gyro) * np.asarray(self.imu_gyro_sign),
            np.asarray(accel) * np.asarray(self.imu_accel_sign),
        )

    def apply_motion(self, motion: np.ndarray) -> np.ndarray:
        """Apply joint signs to an array whose final dimension has eight joints."""

        values = np.asarray(motion)
        if values.ndim == 0 or values.shape[-1] != 8:
            raise ValueError(f"motion must have a final dimension of 8, got {values.shape}")
        return values * np.asarray(self.joint_sign)


def _validated_signs(value: Any, size: int, name: str, default: Sequence[float]) -> tuple[float, ...]:
    """Validate a sign vector while keeping a safe default for omitted YAML fields."""

    if value is None:
        value = default
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise DatasetFormatError(f"{name} must contain numeric signs") from exc
    if len(values) != size or any(not math.isfinite(item) or abs(abs(item) - 1.0) > 1.0e-6 for item in values):
        raise DatasetFormatError(f"{name} must contain exactly {size} values, each +1 or -1")
    return values


def _normalise_sensor_side(value: Any) -> str:
    """Normalize the side labels accepted by the YAML and public API."""

    side = str(value).strip().lower()
    aliases = {
        "b": "base",
        "base": "base",
        "pelvis": "base",
        "torso": "base",
        "waist": "base",
        "l": "left",
        "left": "left",
        "r": "right",
        "right": "right",
    }
    try:
        return aliases[side]
    except KeyError as exc:
        raise DatasetFormatError(f"IMU sensor side must be base, left, or right; got {value!r}") from exc


def _validated_sensor_sides(value: Any, default: Sequence[str] = DEFAULT_IMU_SENSOR_SIDES) -> tuple[str, ...]:
    """Validate the six-sensor side assignment from a calibration file."""

    if value is None:
        value = default
    try:
        sides = tuple(_normalise_sensor_side(item) for item in value)
    except TypeError as exc:
        raise DatasetFormatError("imu_sensor_sides must be a sequence of side labels") from exc
    if len(sides) != len(default):
        raise DatasetFormatError(f"imu_sensor_sides must contain exactly {len(default)} values")
    return sides


def _validated_axis_transforms(
    value: Any,
    default: Mapping[str, tuple[tuple[float, ...], ...]] = DEFAULT_IMU_AXIS_TRANSFORMS,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Validate 3x3 signed-permutation matrices keyed by leg side.

    Matrices use the convention ``output = matrix @ dataset_vector``.  Each
    row and column must contain exactly one ``+1`` or ``-1`` so an accidental
    scale/shear cannot silently corrupt the sensor target.
    """

    if value is None:
        value = default
    if not isinstance(value, Mapping):
        raise DatasetFormatError("imu_axis_transforms must be a mapping of side names to 3x3 matrices")
    merged: dict[str, Any] = dict(default)
    for raw_side, raw_matrix in value.items():
        side = _normalise_sensor_side(raw_side)
        if side == "base":
            raise DatasetFormatError("imu_axis_transforms only accepts left and right leg entries")
        merged[side] = raw_matrix

    validated: dict[str, tuple[tuple[float, ...], ...]] = {}
    for side in ("left", "right"):
        try:
            matrix = np.asarray(merged[side], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(f"imu_axis_transforms[{side!r}] must be numeric") from exc
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise DatasetFormatError(f"imu_axis_transforms[{side!r}] must have finite shape [3, 3]")
        close_to_zero_or_one = np.isclose(np.abs(matrix), 0.0, atol=1.0e-6) | np.isclose(
            np.abs(matrix), 1.0, atol=1.0e-6
        )
        if not close_to_zero_or_one.all():
            raise DatasetFormatError(
                f"imu_axis_transforms[{side!r}] must contain only 0, +1, and -1 values"
            )
        magnitudes = np.where(np.isclose(matrix, 0.0, atol=1.0e-6), 0.0, np.sign(matrix))
        if not np.all(np.sum(np.abs(magnitudes), axis=0) == 1) or not np.all(
            np.sum(np.abs(magnitudes), axis=1) == 1
        ):
            raise DatasetFormatError(
                f"imu_axis_transforms[{side!r}] must be a signed permutation matrix"
            )
        validated[side] = tuple(tuple(float(item) for item in row) for row in magnitudes.tolist())
    return validated


def _validated_rotation_mapping(
    value: Any,
    name: str,
    default: Mapping[str, tuple[tuple[float, ...], ...]],
) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Validate per-sensor right-handed 3-D mounting rotations.

    The source recordings do not contain a sensor-to-link mounting pose, so
    the repository defaults are identity matrices.  Rejecting reflections
    here is deliberate: a reflection belongs in the explicit left/right axis
    convention, not in a quaternion rotation calibration.
    """

    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise DatasetFormatError(f"{name} must be a mapping of IMU IDs to 3x3 rotations")
    merged: dict[str, Any] = dict(default)
    for raw_key, raw_matrix in value.items():
        key = str(raw_key).strip().lower()
        if key in {"default", "all"}:
            keys = tuple(default)
        elif key in default:
            keys = (key,)
        elif key in {"left", "right"}:
            keys = tuple(sensor_id for sensor_id in default if DEFAULT_IMU_SENSOR_SIDES[int(sensor_id[-2:]) - 1] == key)
        else:
            raise DatasetFormatError(f"{name} contains unknown IMU key {raw_key!r}")
        for sensor_id in keys:
            merged[sensor_id] = raw_matrix

    validated: dict[str, tuple[tuple[float, ...], ...]] = {}
    for sensor_id in default:
        try:
            matrix = np.asarray(merged[sensor_id], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(f"{name}[{sensor_id!r}] must be numeric") from exc
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise DatasetFormatError(f"{name}[{sensor_id!r}] must have finite shape [3, 3]")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-5) or not np.isclose(
            np.linalg.det(matrix), 1.0, atol=1.0e-5
        ):
            raise DatasetFormatError(
                f"{name}[{sensor_id!r}] must be a right-handed orthonormal rotation (det=+1)"
            )
        validated[sensor_id] = tuple(tuple(float(item) for item in row) for row in matrix.tolist())
    return validated


def load_direction_calibration(path: str | Path | None = None) -> SensorCoordinateCalibration:
    """Load data-to-simulation direction signs from YAML.

    The root-level file is preferred so an experiment can be versioned beside
    the dataset.  The package config is a fallback for installed/source-tree
    use.  The implicit default falls back to calibrated constants; an explicit
    missing path raises immediately so a typo cannot silently disable a site
    calibration.
    """

    explicit_path = path is not None
    mapping_path = Path(path).expanduser() if explicit_path else _DEFAULT_COORDINATE_MAPPING
    if explicit_path and not mapping_path.is_file():
        raise FileNotFoundError(f"sensor coordinate mapping does not exist: {mapping_path}")
    if not mapping_path.is_file():
        candidates = (_PACKAGE_CONFIG_ROOT / "sensor_coordinate_mapping.yaml",)
        mapping_path = next((candidate for candidate in candidates if candidate.is_file()), mapping_path)
    if not mapping_path.is_file():
        return SensorCoordinateCalibration()
    payload = _yaml_load(mapping_path)
    raw = payload.get("coordinate_calibration", payload)
    if not isinstance(raw, Mapping):
        raise DatasetFormatError("sensor coordinate mapping must contain a 'coordinate_calibration' mapping")
    # ``imu_axis_transform`` was used by an early draft of the mapping file;
    # accept it as a read-only compatibility alias while documenting the
    # plural ``imu_axis_transforms`` spelling for new files.  A legacy file
    # with no matrix section intentionally keeps the old global-sign behavior.
    axis_payload = raw.get("imu_axis_transforms", raw.get("imu_axis_transform"))
    axis_transforms = None if axis_payload is None else _validated_axis_transforms(axis_payload)
    joint_link_rotations = _validated_rotation_mapping(
        raw.get("R_joint_link", raw.get("r_joint_link", raw.get("imu_joint_link_rotations"))),
        "R_joint_link",
        DEFAULT_IMU_JOINT_LINK_ROTATIONS,
    )
    link_sensor_rotations = _validated_rotation_mapping(
        raw.get("R_link_sensor", raw.get("r_link_sensor", raw.get("imu_link_sensor_rotations"))),
        "R_link_sensor",
        DEFAULT_IMU_LINK_SENSOR_ROTATIONS,
    )
    orientation_mode = str(raw.get("orientation_mode", "euler_components")).strip().lower()
    if orientation_mode not in {"euler_components", "rotation_matrix"}:
        raise DatasetFormatError("orientation_mode must be 'euler_components' or 'rotation_matrix'")
    return SensorCoordinateCalibration(
        joint_sign=_validated_signs(raw.get("joint_sign"), 8, "joint_sign", DEFAULT_JOINT_SIGN),
        imu_orientation_sign=_validated_signs(
            raw.get("imu_orientation_sign"), 3, "imu_orientation_sign", DEFAULT_IMU_ORIENTATION_SIGN
        ),
        imu_gyro_sign=_validated_signs(raw.get("imu_gyro_sign"), 3, "imu_gyro_sign", DEFAULT_IMU_GYRO_SIGN),
        imu_accel_sign=_validated_signs(raw.get("imu_accel_sign"), 3, "imu_accel_sign", DEFAULT_IMU_ACCEL_SIGN),
        imu_sensor_sides=_validated_sensor_sides(raw.get("imu_sensor_sides"), DEFAULT_IMU_SENSOR_SIDES),
        imu_axis_transforms=axis_transforms,
        imu_joint_link_rotations=joint_link_rotations,
        imu_link_sensor_rotations=link_sensor_rotations,
        imu_orientation_mode=orientation_mode,
    )


# Backwards-compatible descriptive alias for callers that used the first name
# during development.  New code should use ``load_direction_calibration``.
load_sensor_coordinate_mapping = load_direction_calibration


@dataclass(frozen=True)
class TaskSpec:
    """A task entry from :file:`task_mapping.yaml`."""

    task_id: str
    name: str
    category: str


@dataclass(frozen=True)
class MotionSeries:
    """Optional joint trajectory associated with a trial."""

    t_s: np.ndarray
    joint_pos_rad: np.ndarray
    joint_vel_rad_s: np.ndarray | None = None


@dataclass
class TrialRecord:
    """A discovered trial and its time-aligned non-media channels."""

    trial_id: str
    task_id: str
    subject_id: str
    path: Path
    t_s: np.ndarray
    imu: np.ndarray
    imu_orientation: np.ndarray
    imu_gyro: np.ndarray
    imu_accel: np.ndarray
    imu_valid: np.ndarray
    contact_lr: np.ndarray
    rgb_paths: dict[str, Path]
    event_paths: dict[str, Path]
    motion: MotionSeries | None
    metadata_path: Path | None = None
    event_stream: np.ndarray | None = None

    @property
    def length(self) -> int:
        """Number of synchronized policy-rate samples."""

        return int(self.t_s.shape[0])


def _yaml_load(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - normal Isaac Lab installs include PyYAML
        raise DatasetFormatError("YAML mappings require PyYAML; install pyyaml before loading mappings") from exc
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except OSError as exc:
        raise FileNotFoundError(f"cannot read mapping file: {path}") from exc
    if not isinstance(value, Mapping):
        raise DatasetFormatError(f"mapping file must contain a YAML object: {path}")
    return value


def _yaml_dump(value: Mapping[str, Any], path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise DatasetFormatError("writing YAML mappings requires PyYAML") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(value), stream, allow_unicode=True, sort_keys=False)


def load_task_mapping(path: str | Path | None = None) -> dict[str, TaskSpec]:
    """Load and validate the ten task IDs used by the recorder."""

    mapping_path = Path(path).expanduser() if path is not None else _DEFAULT_TASK_MAPPING
    if not mapping_path.is_file():
        candidates = (
            _REPO_ROOT / "source" / "isaaclab_assets" / "configs" / "task_mapping.yaml",
            _PACKAGE_CONFIG_ROOT / "task_mapping.yaml",
        )
        mapping_path = next((candidate for candidate in candidates if candidate.is_file()), mapping_path)
    payload = _yaml_load(mapping_path)
    raw_tasks = payload.get("tasks", payload)
    if not isinstance(raw_tasks, Mapping):
        raise DatasetFormatError("task mapping must contain a 'tasks' mapping")

    result: dict[str, TaskSpec] = {}
    for task_id, raw in raw_tasks.items():
        task_id = str(task_id).upper()
        if not isinstance(raw, Mapping):
            raise DatasetFormatError(f"task mapping entry {task_id} must be an object")
        name = str(raw.get("name", "")).strip()
        category = str(raw.get("category", "")).strip()
        if not name or not category:
            raise DatasetFormatError(f"task mapping entry {task_id} needs non-empty name and category")
        result[task_id] = TaskSpec(task_id, name, category)

    missing = [task_id for task_id in TASK_IDS if task_id not in result]
    if missing:
        raise DatasetFormatError(f"task mapping is missing task IDs: {', '.join(missing)}")
    return result


def load_imu_mapping(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the six-sensor mapping without resolving it against a USD stage."""

    mapping_path = Path(path).expanduser() if path is not None else _DEFAULT_IMU_MAPPING
    if not mapping_path.is_file():
        candidates = (
            _REPO_ROOT / "source" / "isaaclab_assets" / "configs" / "imu_mapping.yaml",
            _PACKAGE_CONFIG_ROOT / "imu_mapping.yaml",
        )
        mapping_path = next((candidate for candidate in candidates if candidate.is_file()), mapping_path)
    payload = _yaml_load(mapping_path)
    raw = payload.get("imu_mapping", payload)
    if not isinstance(raw, Mapping):
        raise DatasetFormatError("IMU mapping must contain an 'imu_mapping' mapping")
    result: dict[str, dict[str, Any]] = {}
    for sensor_id in IMU_IDS:
        entry = raw.get(sensor_id)
        if not isinstance(entry, Mapping):
            raise DatasetFormatError(f"IMU mapping is missing object entry {sensor_id}")
        if "robot_link" not in entry:
            raise DatasetFormatError(f"IMU mapping entry {sensor_id} has no robot_link")
        result[sensor_id] = dict(entry)
    return result


_LINK_ALIASES: dict[str, tuple[str, ...]] = {
    "imu_01": (
        "left_foot",
        "lfoot",
        "l_foot",
        "foot_left",
        "leftfoot",
        "ankle_left",
        "left_ankle",
    ),
    "imu_02": (
        "right_foot",
        "rfoot",
        "r_foot",
        "foot_right",
        "rightfoot",
        "ankle_right",
        "right_ankle",
    ),
    "imu_03": (
        "left_shank",
        "lleg",
        "l_leg",
        "leftleg",
        "left_shin",
        "shank_left",
        "knee_left",
        "left_knee",
    ),
    "imu_04": (
        "right_shank",
        "rleg",
        "r_leg",
        "rightleg",
        "right_shin",
        "shank_right",
        "knee_right",
        "right_knee",
    ),
    "imu_05": (
        "left_thigh",
        "lthigh",
        "l_thigh",
        "thigh_left",
        "leftthigh",
        "hip_left",
        "left_hip",
    ),
    "imu_06": (
        "right_thigh",
        "rthigh",
        "r_thigh",
        "thigh_right",
        "rightthigh",
        "hip_right",
        "right_hip",
    ),
}


def _normalise_link_name(name: str) -> str:
    """Normalize a USD body path/name for alias matching."""

    token = str(name).replace("\\", "/").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]", "", token)


def resolve_imu_mapping(
    robot_link_names: Iterable[str],
    mapping: Mapping[str, Mapping[str, Any]] | str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, str | None]:
    """Resolve configured IMU links against names discovered in a USD articulation.

    Matching uses exact normalized names followed by the aliases in the YAML.
    Unresolved entries are returned as ``None``.  If ``output_path`` is given,
    a YAML file containing the candidates and unresolved fields is emitted so a
    user can edit it for a robot with non-standard link names.
    """

    if mapping is None or isinstance(mapping, (str, Path)):
        configured = load_imu_mapping(mapping)
    else:
        raw_mapping = mapping.get("imu_mapping", mapping) if isinstance(mapping, Mapping) else mapping
        configured = {str(key): dict(value) for key, value in raw_mapping.items()}
    names = [str(value) for value in robot_link_names]
    normalized = {_normalise_link_name(name): name for name in names}
    resolved: dict[str, str | None] = {}
    candidates: dict[str, list[str]] = {}
    used: set[str] = set()
    for sensor_id in IMU_IDS:
        entry = configured.get(sensor_id, {})
        configured_link = str(entry.get("robot_link", ""))
        aliases = [configured_link, *_LINK_ALIASES.get(sensor_id, ())]
        aliases.extend(str(value) for value in entry.get("aliases", ()) if value is not None)
        match = None
        for alias in aliases:
            candidate = normalized.get(_normalise_link_name(alias))
            if candidate is not None and candidate not in used:
                match = candidate
                break
        if match is None:
            alias_tokens = {_normalise_link_name(alias) for alias in aliases if alias}
            available = [name for token, name in normalized.items() if any(part in token for part in alias_tokens)]
            candidates[sensor_id] = available
        else:
            used.add(match)
        resolved[sensor_id] = match

    if output_path is not None:
        output = {
            "version": 1,
            "imu_mapping": {
                sensor_id: {
                    "sensor_id": index,
                    "robot_link": resolved[sensor_id],
                    "candidates": candidates.get(sensor_id, []),
                }
                for index, sensor_id in enumerate(IMU_IDS, start=1)
            },
            "unresolved": [sensor_id for sensor_id, value in resolved.items() if value is None],
        }
        _yaml_dump(output, Path(output_path).expanduser())
    return resolved


def _parse_time(value: Any) -> float:
    """Parse recorder timestamps, numeric seconds, or ``device+00T...`` strings."""

    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        pass
    # Handles ISO timestamps and the recorder's ``device+00T00:10:00.005``.
    match = re.search(r"(?:T|^)(\d{1,3}):(\d{2}):(\d{2}(?:\.\d+)?)", text)
    if match:
        return float(match.group(1)) * 3600.0 + float(match.group(2)) * 60.0 + float(match.group(3))
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:us|µs)$", text.lower())
    if match:
        return float(match.group(1)) * 1.0e-6
    raise DatasetFormatError(f"cannot parse timestamp '{text}'")


def _column(row: Mapping[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            try:
                return float(value)
            except ValueError:
                return float("nan")
    return float("nan")


def _read_imu_csv(
    path: Path,
    target_dt_s: float,
    calibration: SensorCoordinateCalibration | None = None,
    *,
    sensor_index: int | None = None,
) -> dict[str, np.ndarray]:
    """Read one IMU CSV and map its channels to the simulation frame.

    ``sensor_index`` is the one-based recorder ID.  Supplying it enables the
    left/right leg axis matrices; omitting it retains the legacy global-sign
    behavior for standalone callers and old conversion scripts.
    """

    calibration = calibration or SensorCoordinateCalibration()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise DatasetFormatError(f"IMU CSV has no data rows: {path}")
    columns = tuple(rows[0])
    time_name = next(
        (name for name in ("timestamp_decoded", "timestamp", "t_s", "time_s", "time") if name in columns), None
    )
    times = np.asarray(
        [_parse_time(row.get(time_name)) if time_name else index * target_dt_s for index, row in enumerate(rows)]
    )
    if not np.isfinite(times).any():
        times = np.arange(len(rows), dtype=np.float64) * target_dt_s
    finite_time = np.isfinite(times)
    if not finite_time.all():
        fallback = np.arange(len(rows), dtype=np.float64) * target_dt_s
        times = np.where(finite_time, times, fallback)
        finite_time = np.ones(len(rows), dtype=bool)
    times = times - times[finite_time][0]
    order = np.argsort(times, kind="stable")
    times = times[order]

    def values(*prefixes: str) -> np.ndarray:
        # Match a prefix only at a field boundary.  A plain alias such as
        # ``gy`` must not also match ``gyro_x_dps``/``gyro_y_dps``/``gyro_z_dps``
        # and silently make all three axes read the first column.
        names = [
            name
            for name in columns
            if any(
                name.lower() == prefix.lower()
                or name.lower().startswith(f"{prefix.lower()}_")
                or name.lower().startswith(f"{prefix.lower()}[")
                for prefix in prefixes
            )
        ]
        # Prefer the explicitly named canonical column when available.
        for prefix in prefixes:
            exact = next((name for name in columns if name.lower() == prefix.lower()), None)
            if exact is not None:
                names = [exact]
                break
        data = np.asarray([_column(rows[index], *names) for index in order], dtype=np.float64)
        return data

    acc = np.stack(
        (
            values("acc_x_mps2", "acc_x", "ax"),
            values("acc_y_mps2", "acc_y", "ay"),
            values("acc_z_mps2", "acc_z", "az"),
        ),
        axis=-1,
    )
    gyro = np.stack(
        (
            values("gyro_x_rad_s", "gyro_x_rads", "gyro_x", "gx"),
            values("gyro_y_rad_s", "gyro_y_rads", "gyro_y", "gy"),
            values("gyro_z_rad_s", "gyro_z_rads", "gyro_z", "gz"),
        ),
        axis=-1,
    )
    # The recorder's columns are *_dps.  Generic gyro columns are also dps;
    # only an explicit *_rad_s/rads suffix is treated as SI already.
    if not any("rad" in name.lower() for name in columns if name.lower().startswith("gyro_")):
        gyro *= math.pi / 180.0
    orientation = np.stack(
        (
            values("roll_rad", "roll_rads", "roll_deg", "roll"),
            values("pitch_rad", "pitch_rads", "pitch_deg", "pitch"),
            values("yaw_rad", "yaw_rads", "yaw_deg", "yaw"),
        ),
        axis=-1,
    )
    if not any("rad" in name.lower() for name in columns if name.lower().startswith(("roll", "pitch", "yaw"))):
        orientation *= math.pi / 180.0
    # Apply the frame calibration before interpolation/fill so every sample,
    # including values synthesized for a missing cell, follows one convention.
    orientation, gyro, acc = calibration.apply_imu(
        orientation, gyro, acc, sensor_index=sensor_index
    )
    for array in (acc, gyro, orientation):
        # A malformed cell should not destroy a whole trial.  Hold the nearest
        # finite value, or use zero when an entire channel is absent.
        for channel in range(array.shape[1]):
            finite = np.isfinite(array[:, channel])
            if not finite.any():
                array[:, channel] = 0.0
            elif not finite.all():
                array[:, channel] = np.interp(times, times[finite], array[finite, channel])
    unique, unique_indices = np.unique(times, return_index=True)
    return {
        "t_s": unique,
        "accel": acc[unique_indices].astype(np.float32),
        "gyro": gyro[unique_indices].astype(np.float32),
        "orientation": orientation[unique_indices].astype(np.float32),
    }


def _resample_channels(source_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    result = np.empty((target_t.size, *values.shape[1:]), dtype=np.float32)
    for index in np.ndindex(values.shape[1:]):
        source = values[(slice(None), *index)]
        finite = np.isfinite(source)
        if not finite.any():
            result[(slice(None), *index)] = 0.0
        elif finite.sum() == 1:
            result[(slice(None), *index)] = source[finite][0]
        else:
            result[(slice(None), *index)] = np.interp(
                target_t, source_t[finite], source[finite], left=source[finite][0], right=source[finite][-1]
            )
    return result


def _resolve_path(trial_dir: Path, root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    raw = str(value).replace("\\", "/")
    candidates = [trial_dir / Path(raw).name, trial_dir / raw, root / raw, Path(raw)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    basename = Path(raw).name
    matches = list(trial_dir.glob(basename))
    if matches:
        return matches[0].resolve()
    return None


def _find_metadata(trial_dir: Path) -> Path | None:
    for path in sorted(trial_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping) and ("task" in payload or "files" in payload):
            return path
    return None


def _discover_motion(trial_dir: Path, metadata_path: Path | None) -> Path | None:
    tokens = ("motion", "joint", "kinematic", "q_rad", "qpos")
    for path in sorted(trial_dir.iterdir()):
        if (
            not path.is_file()
            or path == metadata_path
            or path.suffix.lower() not in {".csv", ".json", ".npz", ".pt", ".pth"}
        ):
            continue
        name = path.name.lower()
        if any(token in name for token in tokens) and "imu" not in name and "pressure" not in name:
            return path
    return None


def _motion_mapping(payload: Mapping[str, Any]) -> tuple[Any, Any, Any] | None:
    def pick(*names: str):
        return next((payload[name] for name in names if name in payload), None)

    position = pick("joint_pos_rad", "joint_pos", "q_rad", "q", "joint_position")
    if position is None:
        nested = pick("motion", "target_motion", "kinematics")
        if isinstance(nested, Mapping):
            return _motion_mapping(nested)
        return None
    return pick("t_s", "time_s", "time", "timestamps_s"), position, pick(
        "joint_vel_rad_s", "joint_vel", "qdot_rad_s", "qdot"
    )


def _read_motion(
    path: Path,
    target_dt_s: float,
    joint_sign: Sequence[float] = DEFAULT_JOINT_SIGN,
) -> MotionSeries | None:
    """Read a motion file and apply the canonical joint-direction calibration."""

    joint_sign = _validated_signs(joint_sign, 8, "joint_sign", DEFAULT_JOINT_SIGN)
    suffix = path.suffix.lower()
    payload: Mapping[str, Any] | None = None
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
    elif suffix in {".pt", ".pth"}:
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, Mapping):
            payload = loaded
    elif suffix == ".json":
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, Mapping):
            payload = loaded
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            columns = list(rows[0])
            q_names = []
            for index in range(8):
                q_names.append(
                    next(
                        (
                            name
                            for name in (f"joint_pos_rad[{index}]", f"q_rad_{index}", f"q{index}", f"joint_{index}")
                            if name in columns
                        ),
                        None,
                    )
                )
            if all(name is not None for name in q_names):
                payload = {
                    "t_s": [
                        _parse_time(row.get("t_s", row.get("time_s", row.get("timestamp", index * target_dt_s))))
                        for index, row in enumerate(rows)
                    ],
                    "joint_pos_rad": [[float(row[name]) for name in q_names] for row in rows],
                }
    if payload is None:
        return None
    # A motion file may carry its own calibration metadata.  It overrides the
    # dataset-level default for that trial, which is useful after an encoder
    # replacement while preserving compatibility with older recordings.
    payload_sign = payload.get("joint_sign")
    if payload_sign is None:
        nested = next(
            (
                payload[name]
                for name in ("motion", "target_motion", "kinematics")
                if isinstance(payload.get(name), Mapping)
            ),
            None,
        )
        if isinstance(nested, Mapping):
            payload_sign = nested.get("joint_sign")
    if payload_sign is not None:
        joint_sign = _validated_signs(payload_sign, 8, "joint_sign", joint_sign)
    joint_calibration = SensorCoordinateCalibration(joint_sign=joint_sign)
    values = _motion_mapping(payload)
    if values is None:
        return None
    t_value, q_value, qdot_value = values
    q = np.asarray(q_value, dtype=np.float64)
    if q.ndim == 1:
        q = q[:, None]
    if q.ndim != 2 or q.shape[1] != 8:
        if q.ndim == 2 and q.shape[0] == 8 and q.shape[1] != 8:
            q = q.T
        else:
            raise DatasetFormatError(f"motion joint position must have shape [T, 8]: {path}")
    if t_value is None:
        t = np.arange(q.shape[0], dtype=np.float64) * target_dt_s
    else:
        t = np.asarray([_parse_time(value) for value in np.asarray(t_value).reshape(-1)], dtype=np.float64)
        if t.size != q.shape[0]:
            raise DatasetFormatError(f"motion timestamps and positions have different lengths: {path}")
        t -= t[0]
    finite = np.isfinite(t) & np.isfinite(q).all(axis=1)
    if finite.sum() < 2:
        return None
    t = t[finite]
    q = q[finite]
    order = np.argsort(t, kind="stable")
    t, q = t[order], q[order]
    t, unique_indices = np.unique(t, return_index=True)
    q = joint_calibration.apply_motion(q[unique_indices])
    qdot = None
    if qdot_value is not None:
        qdot = np.asarray(qdot_value, dtype=np.float64)
        if qdot.shape == q.shape:
            qdot = joint_calibration.apply_motion(qdot[finite][order][unique_indices])
        else:
            qdot = None
    if qdot is None:
        qdot = np.gradient(q, t, axis=0, edge_order=1)
    return MotionSeries(t.astype(np.float32), q.astype(np.float32), qdot.astype(np.float32))


def _read_pressure(path: Path, target_dt_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    except OSError:
        return None
    if not rows:
        return None
    columns = list(rows[0])
    t_name = next((name for name in ("timestamp_decoded", "timestamp", "t_s", "time_s") if name in columns), None)
    t = np.asarray([_parse_time(row.get(t_name)) if t_name else i * target_dt_s for i, row in enumerate(rows)])
    finite_t = np.isfinite(t)
    if not finite_t.any():
        t = np.arange(len(rows), dtype=np.float64) * target_dt_s
    else:
        if not finite_t.all():
            fallback = np.arange(len(rows), dtype=np.float64) * target_dt_s
            t = np.where(finite_t, t, fallback)
        t -= t[np.isfinite(t)][0]
    values = []
    for row in rows:
        numbers = []
        for name in columns:
            if name.startswith("p"):
                try:
                    numbers.append(float(row[name]))
                except (TypeError, ValueError):
                    continue
        values.append(np.nanmean(numbers) / 4095.0 if numbers else 0.0)
    return t.astype(np.float64), np.asarray(values, dtype=np.float32)


def _load_event_file(path: Path) -> np.ndarray | None:
    """Load events as ``[N, 4] = (x, y, time_s, polarity)``.

    AEDAT4 support is optional and uses ``dv-processing`` when installed.  A
    preview MP4 is intentionally rejected; deriving events by RGB differencing
    would not represent the asynchronous sensor that was recorded.
    """

    suffix = path.suffix.lower()
    if suffix in {".mp4", ".avi", ".mov"}:
        return None
    events: Any = None
    if suffix in {".npz", ".npy"}:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            events = next((loaded[key] for key in ("events", "event", "data") if key in loaded), None)
        else:
            events = loaded
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            def get(row: Mapping[str, str], *names: str) -> float:
                return _column(row, *names)
            events = np.asarray(
                [
                    [
                        get(row, "x", "event_x"),
                        get(row, "y", "event_y"),
                        get(row, "t_s", "time_s", "timestamp", "t"),
                        get(row, "polarity", "p", "sign"),
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )
    elif suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            events = loaded.get("events", loaded.get("event"))
        else:
            events = loaded
    elif suffix == ".aedat4":
        try:
            import dv_processing as dv  # type: ignore
        except ImportError:
            warnings.warn(
                f"cannot decode AEDAT4 without dv-processing; install it to use events: {path}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        try:
            reader = dv.io.MonoCameraRecording(str(path))
            rows = []
            while reader.isRunning():
                batch = reader.getNextEventBatch()
                if batch is None:
                    continue
                array = batch.numpy()
                if array is not None and len(array):
                    names = array.dtype.names or ()
                    rows.extend(
                        [
                            [item["x"], item["y"], item["timestamp"], item["polarity"]]
                            for item in array
                        ]
                        if names
                        else np.asarray(array).tolist()
                    )
            events = rows
        except Exception as exc:  # pragma: no cover - depends on optional native SDK
            warnings.warn(f"failed to decode AEDAT4 '{path}': {exc}", RuntimeWarning, stacklevel=2)
            return None
    if events is None:
        return None
    array = np.asarray(events, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 4:
        raise DatasetFormatError(f"event stream must have shape [N, 4] or wider: {path}")
    array = array[:, :4]
    finite = np.isfinite(array).all(axis=1)
    array = array[finite]
    if not len(array):
        return np.zeros((0, 4), dtype=np.float32)
    # AEDAT timestamps are commonly microseconds; normalize to seconds.
    magnitude = float(np.nanmax(np.abs(array[:, 2])))
    if magnitude > 1.0e5:
        array[:, 2] *= 1.0e-6
    array[:, 2] -= array[:, 2].min()
    array[:, 3] = np.where(array[:, 3] > 0, 1.0, -1.0)
    return array.astype(np.float32)


class ExoskeletonImitationDataset(Dataset):
    """Time-synchronized windows from the recorded exoskeleton trials.

    Parameters are intentionally independent of Isaac Lab.  ``task_id`` may be
    one ID, an iterable of IDs, or ``None`` for all tasks.  With
    ``random_task_sampling=True`` each item samples a task/trial uniformly;
    otherwise indexing walks through the discovered trials.  Optional media is
    returned as zero-filled tensors plus validity flags when a decoder is not
    installed, which keeps PyTorch collation deterministic.

    ``calibration_path`` selects the YAML sign map used for IMU axes and the
    eight canonical joint channels.  With ``None`` the repository default is
    loaded automatically.

    A returned event stream has ``target_event_stream[N, 4]`` columns
    ``(x, y, t_relative_s, polarity)``; ``target_event_frame_index[N]`` maps
    each event to its policy-rate window bin, and ``target_event`` is the
    fixed-size ``[window, 2 cameras, 2 polarities, H, W]`` histogram used by
    PPO.  Summing the camera axis produces a combined two-polarity histogram.
    ``event_source_size`` uses ``(height, width)`` and defaults to the
    320x240 DVXplorer sensor used by the supplied recordings.
    """

    def __init__(
        self,
        root: str | Path,
        task_id: str | Sequence[str] | None = None,
        *,
        window_size: int = 1,
        target_dt_s: float = 0.02,
        random_window: bool = True,
        random_task_sampling: bool = False,
        steps_per_epoch: int | None = None,
        seed: int = 0,
        rgb_size: tuple[int, int] = (256, 256),
        event_size: tuple[int, int] = (64, 64),
        event_source_size: tuple[int, int] | None = (240, 320),
        load_rgb: bool = True,
        load_event: bool = True,
        strict: bool = False,
        task_mapping_path: str | Path | None = None,
        calibration_path: str | Path | None = None,
    ) -> None:
        if target_dt_s <= 0.0:
            raise ValueError("target_dt_s must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if len(rgb_size) != 2 or min(rgb_size) <= 0 or len(event_size) != 2 or min(event_size) <= 0:
            raise ValueError("rgb_size and event_size must contain positive (height, width)")
        if event_source_size is not None and (len(event_source_size) != 2 or min(event_source_size) <= 0):
            raise ValueError("event_source_size must be None or positive (height, width)")
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"imitation dataset path does not exist: {self.root}")
        self.window_size = int(window_size)
        self.target_dt_s = float(target_dt_s)
        self.random_window = bool(random_window)
        self.random_task_sampling = bool(random_task_sampling)
        self.steps_per_epoch = steps_per_epoch
        self.seed = int(seed)
        self.epoch = 0
        self.rgb_size = tuple(int(value) for value in rgb_size)
        self.event_size = tuple(int(value) for value in event_size)
        self.event_source_size = (
            None if event_source_size is None else tuple(int(value) for value in event_source_size)
        )
        self.load_rgb = bool(load_rgb)
        self.load_event = bool(load_event)
        self.strict = bool(strict)
        self.direction_calibration = load_direction_calibration(calibration_path)
        self.task_mapping = load_task_mapping(task_mapping_path)
        self.requested_task_ids = self._normalise_task_ids(task_id)
        self.trials = self._discover_trials()
        if not self.trials:
            raise DatasetFormatError(f"no usable trials found below {self.root}")
        self._video_readers: dict[Path, Any] = {}
        self._video_captures: dict[Path, Any] = {}
        self._event_cache: dict[Path, np.ndarray | None] = {}
        self._epoch_cache: dict[int, list[tuple[int, int]]] = {}
        # ``sample()`` is called from environment resets rather than through a
        # PyTorch DataLoader, so an internal counter is needed to avoid drawing
        # the identical seeded window on every reset.
        self._sample_counter = 0

    def _normalise_task_ids(self, task_id: str | Sequence[str] | None) -> tuple[str, ...]:
        if task_id is None or str(task_id).lower() in {"", "all", "none"}:
            return TASK_IDS
        if isinstance(task_id, str):
            values = (task_id,)
        else:
            values = tuple(task_id)
        result = tuple(str(value).upper() for value in values)
        unknown = [value for value in result if value not in self.task_mapping]
        if unknown:
            raise ValueError(f"unknown task IDs: {', '.join(unknown)}")
        return result

    def _discover_trials(self) -> list[TrialRecord]:
        if self.root.is_file():
            raise DatasetFormatError("dataset root must be a directory containing task trials")
        trial_dirs = sorted({path for path in self.root.rglob("Trial*") if path.is_dir()})
        result: list[TrialRecord] = []
        for trial_dir in trial_dirs:
            metadata_path = _find_metadata(trial_dir)
            payload: Mapping[str, Any] = {}
            if metadata_path is not None:
                try:
                    loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, Mapping):
                        payload = loaded
                except (OSError, json.JSONDecodeError):
                    pass
            task_id = str(payload.get("task", ""))
            if not task_id:
                task_id = next((part for part in reversed(trial_dir.parts) if part in TASK_IDS), "")
            task_id = task_id.upper()
            if task_id not in self.requested_task_ids:
                continue
            if task_id not in TASK_IDS:
                if self.strict:
                    raise DatasetFormatError(f"cannot infer a supported task ID for {trial_dir}")
                continue
            subject = payload.get("subject")
            subject_id = str(subject.get("uid", "")) if isinstance(subject, Mapping) else ""
            subject_id = subject_id or next(
                (part for part in trial_dir.parts if re.fullmatch(r"P\d+", part.upper())), "unknown"
            )
            try:
                record = self._load_trial(trial_dir, task_id, subject_id, metadata_path, payload)
            except (DatasetFormatError, OSError) as exc:
                if self.strict:
                    raise
                _LOGGER.warning("Skipping trial %s: %s", trial_dir, exc)
                continue
            if record.length >= 1:
                result.append(record)
        return result

    def _load_trial(
        self,
        trial_dir: Path,
        task_id: str,
        subject_id: str,
        metadata_path: Path | None,
        payload: Mapping[str, Any],
    ) -> TrialRecord:
        files = payload.get("files", {}) if isinstance(payload.get("files"), Mapping) else {}
        imu_sources: list[dict[str, np.ndarray] | None] = []
        for index in range(1, 7):
            key = f"imu_{index:02d}_csv"
            path = _resolve_path(trial_dir, self.root, files.get(key))
            if path is None:
                matches = sorted(trial_dir.glob(f"*_imu_{index:02d}.csv"))
                path = matches[0] if matches else None
            imu_sources.append(
                _read_imu_csv(
                    path,
                    self.target_dt_s,
                    self.direction_calibration,
                    sensor_index=index,
                )
                if path is not None
                else None
            )
        available = [source for source in imu_sources if source is not None]
        if not available:
            raise DatasetFormatError(f"trial has no imu_01..imu_06 CSV files: {trial_dir}")
        start = max(float(source["t_s"][0]) for source in available)
        end = min(float(source["t_s"][-1]) for source in available)
        if end <= start:
            longest = max(available, key=lambda source: source["t_s"].size)
            start, end = float(longest["t_s"][0]), float(longest["t_s"][-1])
        count = max(1, int(math.floor((end - start) / self.target_dt_s + 1.0e-6)) + 1)
        target_t = np.arange(count, dtype=np.float64) * self.target_dt_s
        target_absolute = target_t + start
        orientations = []
        gyros = []
        accels = []
        valid = []
        for source in imu_sources:
            if source is None:
                orientations.append(np.zeros((count, 3), dtype=np.float32))
                gyros.append(np.zeros((count, 3), dtype=np.float32))
                accels.append(np.zeros((count, 3), dtype=np.float32))
                valid.append(False)
                continue
            orientation = _resample_channels(source["t_s"], source["orientation"], target_absolute)
            gyro = _resample_channels(source["t_s"], source["gyro"], target_absolute)
            accel = _resample_channels(source["t_s"], source["accel"], target_absolute)
            orientations.append(orientation)
            gyros.append(gyro)
            accels.append(accel)
            valid.append(True)
        orientation_array = np.stack(orientations, axis=1)
        gyro_array = np.stack(gyros, axis=1)
        accel_array = np.stack(accels, axis=1)
        imu = np.concatenate((orientation_array, gyro_array, accel_array), axis=-1).astype(np.float32)

        contact = np.zeros((count, 2), dtype=np.float32)
        pressure_paths = []
        for side in ("left", "right"):
            value = files.get(f"pressure_{side}_csv")
            path = _resolve_path(trial_dir, self.root, value)
            if path is None:
                matches = sorted(trial_dir.glob(f"*pressure_{side}.csv"))
                path = matches[0] if matches else None
            pressure_paths.append(path)
        for side, path in enumerate(pressure_paths):
            if path is not None:
                pressure = _read_pressure(path, self.target_dt_s)
                if pressure is not None:
                    pressure_t, pressure_values = pressure
                    contact[:, side] = _resample_channels(
                        pressure_t, pressure_values[:, None], target_absolute
                    )[:, 0].clip(0.0, 1.0)

        rgb_paths: dict[str, Path] = {}
        event_paths: dict[str, Path] = {}
        for side in ("left", "right"):
            rgb = _resolve_path(trial_dir, self.root, files.get(f"{side}_rgb"))
            if rgb is None:
                matches = sorted(trial_dir.glob(f"{side}_*recording*.mp4"))
                if matches:
                    rgb = matches[0]
            if rgb is not None:
                rgb_paths[side] = rgb
            event = _resolve_path(trial_dir, self.root, files.get(f"{side}_event"))
            # Metadata points to *_event_preview.mp4.  Prefer the original AEDAT4.
            event_candidates = [path for path in trial_dir.glob(f"{side}_*.aedat4")]
            if event_candidates:
                event = sorted(event_candidates)[0]
            if event is None:
                event_candidates = [
                    path
                    for suffix in (".npz", ".npy", ".csv", ".json")
                    for path in trial_dir.glob(f"{side}_*event*{suffix}")
                ]
                if event_candidates:
                    event = sorted(event_candidates)[0]
            if event is not None and event.suffix.lower() in {".aedat4", ".npz", ".npy", ".csv", ".json"}:
                event_paths[side] = event

        motion_path = _discover_motion(trial_dir, metadata_path)
        motion = None
        if motion_path is not None:
            try:
                motion = _read_motion(
                    motion_path,
                    self.target_dt_s,
                    joint_sign=self.direction_calibration.joint_sign,
                )
            except DatasetFormatError as exc:
                _LOGGER.warning("Ignoring malformed optional motion file %s: %s", motion_path, exc)
        if motion is not None:
            motion = self._resample_motion(motion, target_absolute)
        return TrialRecord(
            trial_id=trial_dir.name,
            task_id=task_id,
            subject_id=subject_id,
            path=trial_dir,
            t_s=target_t.astype(np.float32),
            imu=imu,
            imu_orientation=orientation_array,
            imu_gyro=gyro_array,
            imu_accel=accel_array,
            imu_valid=np.asarray(valid, dtype=bool),
            contact_lr=contact,
            rgb_paths=rgb_paths,
            event_paths=event_paths,
            motion=motion,
            metadata_path=metadata_path,
        )

    @staticmethod
    def _resample_motion(motion: MotionSeries, target_t: np.ndarray) -> MotionSeries:
        position = _resample_channels(motion.t_s, motion.joint_pos_rad, target_t)
        velocity = (
            None
            if motion.joint_vel_rad_s is None
            else _resample_channels(motion.t_s, motion.joint_vel_rad_s, target_t)
        )
        return MotionSeries(target_t.astype(np.float32), position, velocity)

    def set_epoch(self, epoch: int) -> None:
        """Change the deterministic random stream used by random windows."""

        self.epoch = int(epoch)
        self._sample_counter = 0

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Task IDs represented by discovered trials."""

        return tuple(sorted({record.task_id for record in self.trials}))

    def __len__(self) -> int:
        if self.steps_per_epoch is not None:
            return max(1, int(self.steps_per_epoch))
        if self.random_window or self.random_task_sampling:
            return max(1, len(self.trials))
        return sum(max(1, record.length - self.window_size + 1) for record in self.trials)

    def _index_to_trial_start(self, index: int) -> tuple[int, int]:
        if self.random_task_sampling:
            rng = random.Random(self.seed + self.epoch * 1_000_003 + int(index))
            trial_index = rng.randrange(len(self.trials))
            record = self.trials[trial_index]
            start = rng.randrange(max(1, record.length - self.window_size + 1)) if self.random_window else 0
            return trial_index, start
        if self.random_window:
            trial_index = int(index) % len(self.trials)
            record = self.trials[trial_index]
            rng = random.Random(self.seed + self.epoch * 1_000_003 + int(index))
            return trial_index, rng.randrange(max(1, record.length - self.window_size + 1))
        remaining = int(index)
        for trial_index, record in enumerate(self.trials):
            count = max(1, record.length - self.window_size + 1)
            if remaining < count:
                return trial_index, remaining
            remaining -= count
        return len(self.trials) - 1, max(0, self.trials[-1].length - self.window_size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trial_index, start = self._index_to_trial_start(index)
        return self._sample_record(self.trials[trial_index], start)

    def sample(
        self,
        task_id: str | None = None,
        timestamp_s: float | None = None,
        *,
        window_size: int | None = None,
        trial_id: str | None = None,
        random_window: bool | None = None,
    ) -> dict[str, Any]:
        """Sample a task/timestamp window for environment-side playback."""

        requested = None if task_id is None else str(task_id).upper()
        candidates = [record for record in self.trials if requested is None or record.task_id == requested]
        if trial_id is not None:
            candidates = [record for record in candidates if record.trial_id == trial_id]
        if not candidates:
            raise KeyError(f"no trial matches task_id={task_id!r}, trial_id={trial_id!r}")
        # Include a monotonically increasing draw index.  This remains
        # reproducible for a fixed seed/epoch while ensuring independent reset
        # calls do not reuse one frozen target window.
        draw_index = self._sample_counter
        self._sample_counter += 1
        rng = random.Random(self.seed + self.epoch * 1_000_003 + draw_index)
        if self.random_task_sampling and requested is None:
            by_task: dict[str, list[TrialRecord]] = {}
            for candidate in candidates:
                by_task.setdefault(candidate.task_id, []).append(candidate)
            task_pool = sorted(by_task)
            task = task_pool[rng.randrange(len(task_pool))]
            candidates = by_task[task]
        record = candidates[rng.randrange(len(candidates))]
        size = self.window_size if window_size is None else int(window_size)
        if size <= 0:
            raise ValueError("window_size must be positive")
        if timestamp_s is None:
            use_random = self.random_window if random_window is None else bool(random_window)
            start = rng.randrange(max(1, record.length - size + 1)) if use_random else 0
        else:
            start = int(np.searchsorted(record.t_s, float(timestamp_s), side="left"))
            start = min(max(start, 0), max(0, record.length - size))
        return self._sample_record(record, start, size)

    def sample_at(self, task_id: str, timestamp_s: float, window_size: int | None = None) -> dict[str, Any]:
        """Alias for timestamp-addressed sampling used by training commands."""

        return self.sample(task_id, timestamp_s, window_size=window_size, random_window=False)

    def _sample_record(self, record: TrialRecord, start: int, size: int | None = None) -> dict[str, Any]:
        size = self.window_size if size is None else int(size)
        if size <= 0:
            raise ValueError("window_size must be positive")
        indices = np.minimum(np.arange(start, start + size), max(0, record.length - 1))
        t_window = record.t_s[indices]
        target_motion = np.zeros((size, 8), dtype=np.float32)
        target_velocity = np.zeros((size, 8), dtype=np.float32)
        motion_valid = False
        if record.motion is not None:
            motion_indices = np.minimum(indices, record.motion.joint_pos_rad.shape[0] - 1)
            target_motion[:] = record.motion.joint_pos_rad[motion_indices]
            if record.motion.joint_vel_rad_s is not None:
                target_velocity[:] = record.motion.joint_vel_rad_s[motion_indices]
            motion_valid = True

        target_rgb, rgb_valid = self._sample_rgb(record, t_window)
        target_event, event_stream, event_frame_index, event_camera_index, event_valid = self._sample_events(
            record, float(t_window[0]), float(t_window[-1] + self.target_dt_s), size
        )
        return {
            "task_id": record.task_id,
            "task_name": self.task_mapping[record.task_id].name,
            "subject_id": record.subject_id,
            "trial_id": record.trial_id,
            "t_s": torch.as_tensor(t_window, dtype=torch.float32),
            "target_motion": torch.as_tensor(target_motion),
            "target_joint_velocity": torch.as_tensor(target_velocity),
            "motion_valid": torch.tensor(motion_valid),
            "target_imu": torch.as_tensor(record.imu[indices]),
            "target_imu_orientation": torch.as_tensor(record.imu_orientation[indices]),
            "target_imu_gyro": torch.as_tensor(record.imu_gyro[indices]),
            "target_imu_accel": torch.as_tensor(record.imu_accel[indices]),
            "imu_valid": torch.as_tensor(record.imu_valid),
            "contact_lr": torch.as_tensor(record.contact_lr[indices]),
            "target_rgb": target_rgb,
            "rgb_valid": rgb_valid,
            "target_event": target_event,
            "target_event_stream": event_stream,
            "target_event_frame_index": event_frame_index,
            "target_event_camera_index": event_camera_index,
            "event_valid": torch.tensor(event_valid),
        }

    def _sample_rgb(self, record: TrialRecord, t_window: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = self.rgb_size
        output = torch.zeros((len(t_window), 2, 3, height, width), dtype=torch.float32)
        valid = torch.zeros(2, dtype=torch.bool)
        if not self.load_rgb:
            return output, valid
        for side_index, side in enumerate(("left", "right")):
            path = record.rgb_paths.get(side)
            if path is None:
                continue
            frames = []
            for time_s in t_window:
                frame = self._read_video_frame(path, float(time_s))
                if frame is None:
                    frames = []
                    break
                frames.append(frame)
            if frames:
                output[:, side_index] = torch.stack(frames)
                valid[side_index] = True
        return output, valid

    def _read_video_frame(self, path: Path, time_s: float) -> torch.Tensor | None:
        try:
            if path.suffix.lower() in {".npy", ".npz"}:
                loaded = np.load(path, allow_pickle=False)
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    array = next((loaded[key] for key in ("frames", "rgb", "images") if key in loaded), None)
                else:
                    array = loaded
                if array is None or array.ndim < 3:
                    return None
                index = min(int(round(time_s / self.target_dt_s)), array.shape[0] - 1) if array.ndim == 4 else 0
                image = array[index] if array.ndim == 4 else array
            else:
                try:
                    import imageio.v2 as imageio
                    reader = self._video_readers.get(path)
                    if reader is None:
                        reader = imageio.get_reader(str(path))
                        self._video_readers[path] = reader
                    metadata = reader.get_meta_data()
                    fps = float(metadata.get("fps", 30.0) or 30.0)
                    image = reader.get_data(max(0, int(round(time_s * fps))))
                except Exception as imageio_exc:  # optional ffmpeg/PyAV decoder
                    # OpenCV is a common Isaac/ROS dependency and can decode
                    # the supplied MP4 files even when imageio's ffmpeg plugin
                    # is absent.  It is a decoder fallback only; event files
                    # never pass through this path.
                    try:
                        import cv2

                        capture = self._video_captures.get(path)
                        if capture is None:
                            capture = cv2.VideoCapture(str(path))
                            if not capture.isOpened():
                                capture.release()
                                raise RuntimeError("OpenCV could not open the video")
                            self._video_captures[path] = capture
                        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
                        ok, bgr = capture.read()
                        if not ok or bgr is None:
                            raise RuntimeError("OpenCV returned no frame")
                        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    except Exception as cv2_exc:
                        _LOGGER.debug(
                            "RGB frame decoder unavailable for %s (imageio=%s, opencv=%s)",
                            path,
                            imageio_exc,
                            cv2_exc,
                        )
                        return None
            image = np.asarray(image)
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            if image.shape[-1] == 4:
                image = image[..., :3]
            if image.shape[-1] != 3:
                return None
            from PIL import Image

            image = np.asarray(Image.fromarray(image.astype(np.uint8)).resize((self.rgb_size[1], self.rgb_size[0])))
            return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        except Exception as exc:
            _LOGGER.warning("failed to read RGB frame %s: %s", path, exc)
            return None

    def _sample_events(
        self, record: TrialRecord, start_s: float, end_s: float, window_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        height, width = self.event_size
        histogram = torch.zeros((window_size, 2, 2, height, width), dtype=torch.float32)
        if not self.load_event or not record.event_paths:
            return (
                histogram,
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                False,
            )
        arrays = []
        for camera_index, side in enumerate(("left", "right")):
            path = record.event_paths.get(side)
            if path is None:
                continue
            if path not in self._event_cache:
                self._event_cache[path] = _load_event_file(path)
            if self._event_cache[path] is not None:
                arrays.append((camera_index, self._event_cache[path]))
        if not arrays:
            return (
                histogram,
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                False,
            )
        streams = []
        frame_indices = []
        camera_indices = []
        for camera_index, events in arrays:
            selected = events[(events[:, 2] >= start_s) & (events[:, 2] < end_s)]
            if not len(selected):
                continue
            selected = selected.copy()
            selected[:, 2] -= start_s
            frame_index = np.floor(selected[:, 2] / self.target_dt_s).astype(np.int64).clip(0, window_size - 1)
            x, y = self._event_grid_coordinates(selected[:, 0], selected[:, 1], width, height)
            polarity = (selected[:, 3] > 0).astype(np.int64)
            for frame, xx, yy, pol in zip(frame_index, x, y, polarity, strict=True):
                histogram[frame, camera_index, pol, yy, xx] += 1.0
            streams.append(selected)
            frame_indices.append(frame_index)
            camera_indices.append(np.full(len(selected), camera_index, dtype=np.int64))
        if streams:
            stream = torch.as_tensor(np.concatenate(streams, axis=0), dtype=torch.float32)
            event_frame_index = torch.as_tensor(np.concatenate(frame_indices), dtype=torch.long)
            event_camera_index = torch.as_tensor(np.concatenate(camera_indices), dtype=torch.long)
        else:
            stream = torch.zeros((0, 4), dtype=torch.float32)
            event_frame_index = torch.zeros(0, dtype=torch.long)
            event_camera_index = torch.zeros(0, dtype=torch.long)
        # A successfully decoded DVS recording is valid even when this short
        # window contains no threshold crossings.  A zero-event interval is
        # an observable quiet scene, whereas ``False`` is reserved for a
        # missing/unsupported/corrupt event stream (handled above).
        return histogram, stream, event_frame_index, event_camera_index, True

    def _event_grid_coordinates(
        self, x: np.ndarray, y: np.ndarray, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map source-sensor pixels to the fixed PPO event grid.

        DVXplorer recordings in the supplied dataset use 320x240 pixels.  The
        target histogram is normally 64x64, so clipping raw coordinates would
        collapse almost all events onto the last row/column.  ``None`` opts
        into conservative inference from the observed coordinate extent; the
        default keeps a stable calibration even when a short window does not
        visit every source pixel.
        """

        if self.event_source_size is None:
            source_height = max(1, int(np.nanmax(y)) + 1) if len(y) else height
            source_width = max(1, int(np.nanmax(x)) + 1) if len(x) else width
        else:
            source_height, source_width = self.event_source_size
        scaled_x = np.floor(np.asarray(x, dtype=np.float64) * width / source_width).astype(np.int64)
        scaled_y = np.floor(np.asarray(y, dtype=np.float64) * height / source_height).astype(np.int64)
        return scaled_x.clip(0, width - 1), scaled_y.clip(0, height - 1)

    def close(self) -> None:
        """Release open video readers."""

        for reader in self._video_readers.values():
            with contextlib.suppress(Exception):
                reader.close()
        self._video_readers.clear()
        for capture in self._video_captures.values():
            with contextlib.suppress(Exception):
                capture.release()
        self._video_captures.clear()

    def __del__(self):  # pragma: no cover - interpreter shutdown order is undefined
        with contextlib.suppress(Exception):
            self.close()


__all__ = [
    "DEFAULT_IMU_ACCEL_SIGN",
    "DEFAULT_IMU_AXIS_TRANSFORMS",
    "DEFAULT_IMU_GYRO_SIGN",
    "DEFAULT_IMU_JOINT_LINK_ROTATIONS",
    "DEFAULT_IMU_LINK_SENSOR_ROTATIONS",
    "DEFAULT_IMU_ORIENTATION_SIGN",
    "DEFAULT_IMU_SENSOR_SIDES",
    "DEFAULT_JOINT_SIGN",
    "DatasetFormatError",
    "ExoskeletonImitationDataset",
    "IMU_IDS",
    "SensorCoordinateCalibration",
    "TASK_IDS",
    "TaskSpec",
    "TrialRecord",
    "load_direction_calibration",
    "load_imu_mapping",
    "load_sensor_coordinate_mapping",
    "load_task_mapping",
    "resolve_imu_mapping",
]
