# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""外骨骼视觉、IMU 与关节运动模仿的数据读取层。

本项目的记录器把一次试验（trial）保存为下面的目录结构::

    records/<subject>/<date>/<task-id>/TrialXX/

每个 trial 通常包含 ``*_imu_01.csv`` 到 ``*_imu_11.csv``、左右两路 RGB 视频和
两路 DVXplorer ``.aedat4``。训练只使用 01--06 六个腿部 IMU。创建数据集时会立即
读取体积较小的 IMU、压力和可选关节轨迹，并统一到 ``target_dt_s`` 时间轴；视频帧
与事件流体积大，只在某个采样窗口真正需要时延迟解码和缓存。事件预览 MP4 绝不会
冒充异步事件数据。

输出是适合 PyTorch/DataLoader 和仿真环境使用的字典。每个可选模态同时带 ``valid``
掩码：零张量只用于保持批处理 shape 稳定，``False`` 才表示文件缺失、损坏或解码器
不可用。模块不导入 Isaac Sim，因此数据预处理和单元测试无需启动 Kit。

读取顺序为：加载任务/坐标 YAML -> 发现 trial -> 解析并校准非媒体通道 -> 按需抽取
时间窗口 -> 延迟读取 RGB/Event。默认 ``sensor_coordinate_mapping.yaml`` 会把四个
矢状面髋/膝通道及左右 IMU 坐标映射到仿真关节约定；实体安装方向不同的场地应通过
``calibration_path`` 提供实测标定，不能靠修改奖励掩盖方向错误。
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
    """表示记录文件不满足可安全解释的数据契约。"""


# 记录仪与 URDF 都采用右腿在前、左腿在后的顺序。髋/膝轴关于矢状面镜像，因此数据到
# articulation 的符号需要区分左右侧：右髋 ``+``、右膝 ``-``、左髋 ``-``、左膝
# ``+``。转向与踝关节的编码器约定无法只根据关节轴推断，所以仍允许配置；完成单关节
# 标定试验后，可用现场专属 YAML 覆盖此元组。
DEFAULT_JOINT_SIGN = (1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0)
# 这些是未指定记录仪传感器的旧调用方所用兼容符号。六通道数据集路径使用下方显式
# 左/右矩阵，因此不会再次应用这些向量。
DEFAULT_IMU_ACCEL_SIGN = (-1.0, -1.0, 1.0)
DEFAULT_IMU_GYRO_SIGN = (-1.0, -1.0, 1.0)
DEFAULT_IMU_ORIENTATION_SIGN = (1.0, -1.0, 1.0)
# 记录仪保留的通道按左右腿交错：IMU-01/03/05 安装在左腿，IMU-02/04/06 安装在右腿；
# 其中 01/02 是踝部、03/04 是膝部、05/06 是髋部。模仿加载器有意忽略 IMU-07..11。
# 矩阵每一行表示一个仿真轴在数据集坐标中的表达；例如右腿矩阵把
# ``(dataset_x, dataset_y, dataset_z)`` 映射为 ``(joint_y, joint_x, -joint_z)``。
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
DEFAULT_IMU_JOINT_LINK_ROTATIONS = {f"imu_{index:02d}": _IDENTITY_ROTATION for index in range(1, 7)}
DEFAULT_IMU_LINK_SENSOR_ROTATIONS = {f"imu_{index:02d}": _IDENTITY_ROTATION for index in range(1, 7)}


@dataclass(frozen=True)
class SensorCoordinateCalibration:
    """把记录仪坐标和关节方向转换到仿真约定。

    左右腿使用显式的带符号置换矩阵处理镜像安装，转换只改变数值方向，不改变数组
    shape 或 PPO 输入维度。旧版全局符号仍供未提供 ``sensor_index`` 的外部调用者
    使用；本数据集保留的六路腿部 IMU 默认都走逐传感器矩阵。

    所有 IMU 输入最后一维必须是 3。关节输入最后一维必须是 8，顺序采用
    ``robot_contract.ACTUATED_JOINT_NAMES``。角度使用 rad、角速度使用 rad/s、线加速度
    使用 m/s^2。
    """

    joint_sign: tuple[float, ...] = DEFAULT_JOINT_SIGN
    imu_orientation_sign: tuple[float, ...] = DEFAULT_IMU_ORIENTATION_SIGN
    imu_gyro_sign: tuple[float, ...] = DEFAULT_IMU_GYRO_SIGN
    imu_accel_sign: tuple[float, ...] = DEFAULT_IMU_ACCEL_SIGN
    imu_sensor_sides: tuple[str, ...] = DEFAULT_IMU_SENSOR_SIDES
    imu_axis_transforms: Mapping[str, tuple[tuple[float, ...], ...]] | None = field(
        default_factory=lambda: dict(DEFAULT_IMU_AXIS_TRANSFORMS)
    )
    # 物理安装旋转刻意与旧版带符号轴映射分开：后者包含项目的左右数据约定（左侧矩阵
    # 是反射），而下面两个矩阵表示真实坐标系旋转，必须保持右手系。
    imu_joint_link_rotations: Mapping[str, tuple[tuple[float, ...], ...]] = field(
        default_factory=lambda: dict(DEFAULT_IMU_JOINT_LINK_ROTATIONS)
    )
    imu_link_sensor_rotations: Mapping[str, tuple[tuple[float, ...], ...]] = field(
        default_factory=lambda: dict(DEFAULT_IMU_LINK_SENSOR_ROTATIONS)
    )
    imu_orientation_mode: str = "euler_components"

    def _sensor_side(self, sensor_index: int | None, sensor_side: str | None) -> str | None:
        """把记录仪编号解析为 ``left``、``right`` 或 ``base``。

        Args:
            sensor_index: 一基编号；01/02 是踝、03/04 是膝、05/06 是髋。
            sensor_side: 调用者显式给出的侧别；提供后优先于编号。

        Returns:
            规范化侧别；两项均未提供时返回 ``None``。

        Raises:
            ValueError: 编号不在已配置六路传感器范围内。
            DatasetFormatError: 侧别文本无法识别。
        """

        if sensor_side is not None:
            return _normalise_sensor_side(sensor_side)
        if sensor_index is None:
            return None
        if sensor_index < 1 or sensor_index > len(self.imu_sensor_sides):
            raise ValueError(f"sensor_index must be in [1, {len(self.imu_sensor_sides)}], got {sensor_index}")
        return self.imu_sensor_sides[sensor_index - 1]

    def _apply_axis_transform(self, values: np.ndarray, sensor_side: str | None) -> np.ndarray | None:
        """应用某侧的 3x3 轴映射；需要旧版全局符号时返回 ``None``。

        Args:
            values: shape 为 ``[..., 3]`` 的方向、角速度或加速度数组。
            sensor_side: 规范化后的 ``left``、``right``、``base`` 或 ``None``。

        Returns:
            shape 和 dtype 与输入一致的转换结果；没有逐侧矩阵时返回 ``None``。

        Raises:
            ValueError: 输入最后一维不是 3。
        """

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
        """返回安装旋转表使用的 IMU 键；无法确定具体传感器时返回 ``None``。"""

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
        """组合并返回 ``R_joint_link @ R_link_sensor @ R_sensor_IMU``。

        ``R_sensor_IMU`` 是记录坐标到腿部坐标的正交轴映射；左腿约定的行列式为
        ``-1``，所以它适合重排向量通道，却不能单独当作物理四元数旋转。另外两个
        矩阵是右手系物理安装旋转，未从 USD/夹具或标定试验测量前默认为单位矩阵。

        Returns:
            shape ``[3, 3]`` 的 ``float64`` NumPy 数组；旧版 ``base``/无侧别模式返回
            ``None``。
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
        """把 ``[...,3]`` 的 ``[roll,pitch,yaw]`` [rad] 转成 XYZ 旋转矩阵。"""

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
        """把 XYZ 旋转矩阵转回 ``[...,3]`` 的 ``[roll,pitch,yaw]`` [rad]。"""

        matrix = np.asarray(matrix, dtype=np.float64)
        pitch = np.arcsin(np.clip(-matrix[..., 2, 0], -1.0, 1.0))
        cp = np.cos(pitch)
        roll = np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2])
        yaw = np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
        singular = np.abs(cp) < 1.0e-7
        if np.any(singular):
            # 使用确定性的万向节锁约定：保持 yaw=0，把剩余旋转归入 roll。
            roll = np.where(singular, np.arctan2(-matrix[..., 0, 1], matrix[..., 1, 1]), roll)
            yaw = np.where(singular, 0.0, yaw)
        return np.stack((roll, pitch, yaw), axis=-1)

    def _transform_orientation(self, orientation: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """按配置模式变换 ``[...,3]`` 欧拉角 [rad]，并保持输入 dtype。"""

        values = np.asarray(orientation)
        if self.imu_orientation_mode == "euler_components":
            # 这是记录仪已发布欧拉角通道的兼容模式，采用既定的分量约定而非四元数相乘；
            # 完成安装标定后应使用 rotation_matrix。
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
        """返回完成坐标标定的 ``(orientation, gyro, accel)`` 数组。

        ``sensor_index`` 与文件名 ``imu_01``--``imu_06`` 的一基后缀一致。对腿部 IMU，
        姿态分量、陀螺仪和加速度使用同一坐标变换；省略编号会保留外部旧调用者所需的
        全局符号行为。

        Args:
            orientation: shape ``[..., 3]`` 的 roll/pitch/yaw [rad]。
            gyro: shape ``[..., 3]`` 的角速度 [rad/s]。
            accel: shape ``[..., 3]`` 的线加速度 [m/s^2]。
            sensor_index: 可选一基记录仪编号。
            sensor_side: 可选侧别，显式提供时优先。

        Returns:
            三个 shape 与各自输入相同的 NumPy 数组。

        Raises:
            DatasetFormatError: 旋转矩阵模式收到非右手系有效变换。
            ValueError: 传感器编号或输入 shape 不合法。
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
        """把八关节记录值的正负方向转换为 canonical 仿真方向。

        Args:
            motion: shape ``[..., 8]`` 的关节角 [rad] 或关节速度 [rad/s]。

        Returns:
            与输入 shape 相同的 NumPy 数组。

        Raises:
            ValueError: 最后一维不是八个关节。
        """

        values = np.asarray(motion)
        if values.ndim == 0 or values.shape[-1] != 8:
            raise ValueError(f"motion must have a final dimension of 8, got {values.shape}")
        return values * np.asarray(self.joint_sign)


def _validated_signs(value: Any, size: int, name: str, default: Sequence[float]) -> tuple[float, ...]:
    """校验只含 ``+1/-1`` 的符号向量；YAML 省略字段时使用安全默认值。"""

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
    """把 YAML/公开 API 接受的侧别别名统一为 base、left 或 right。"""

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
    """校验标定文件中的六路传感器侧别分配。"""

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
    """校验按左右腿索引的 3x3 带符号置换矩阵。

    约定为 ``output = matrix @ dataset_vector``。每行每列必须恰有一个 ``+1/-1``，
    从而拒绝会悄悄缩放或剪切传感器目标的错误矩阵。
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
            raise DatasetFormatError(f"imu_axis_transforms[{side!r}] must contain only 0, +1, and -1 values")
        magnitudes = np.where(np.isclose(matrix, 0.0, atol=1.0e-6), 0.0, np.sign(matrix))
        if not np.all(np.sum(np.abs(magnitudes), axis=0) == 1) or not np.all(np.sum(np.abs(magnitudes), axis=1) == 1):
            raise DatasetFormatError(f"imu_axis_transforms[{side!r}] must be a signed permutation matrix")
        validated[side] = tuple(tuple(float(item) for item in row) for row in magnitudes.tolist())
    return validated


def _validated_rotation_mapping(
    value: Any,
    name: str,
    default: Mapping[str, tuple[tuple[float, ...], ...]],
) -> dict[str, tuple[tuple[float, ...], ...]]:
    """校验每路 IMU 的右手系三维物理安装旋转。

    源记录不含 sensor-to-link 安装姿态，所以默认单位矩阵。这里有意拒绝反射：镜像
    应写入显式左右轴约定，不能混入可由四元数表示的物理旋转标定。
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
            raise DatasetFormatError(f"{name}[{sensor_id!r}] must be a right-handed orthonormal rotation (det=+1)")
        validated[sensor_id] = tuple(tuple(float(item) for item in row) for row in matrix.tolist())
    return validated


def load_direction_calibration(path: str | Path | None = None) -> SensorCoordinateCalibration:
    """从 YAML 加载记录坐标到仿真坐标的标定。

    未给路径时优先使用仓库根目录文件，再回退到包内配置，最后回退到代码中的默认
    常量。显式路径不存在则立即报错，避免拼写错误悄悄关闭场地标定。

    Args:
        path: 可选 ``sensor_coordinate_mapping.yaml`` 路径。

    Returns:
        经过数值、shape、正交性和左右侧别校验的标定对象。

    Raises:
        FileNotFoundError: 显式指定的文件不存在或不可读取。
        DatasetFormatError: YAML 结构、符号、3x3 轴矩阵或安装旋转不合法。
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
    # 映射文件早期草案使用 ``imu_axis_transform``；这里把它作为只读兼容别名接受，新文件
    # 则统一记录复数形式 ``imu_axis_transforms``。没有矩阵段的旧文件会保留原全局符号行为。
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


# 这是为开发期间使用旧名称的调用方保留的向后兼容别名；新代码应使用
# ``load_direction_calibration`` 函数。
load_sensor_coordinate_mapping = load_direction_calibration


@dataclass(frozen=True)
class TaskSpec:
    """``task_mapping.yaml`` 中一个 BT/AT 任务的编号、名称和类别。"""

    task_id: str
    name: str
    category: str


@dataclass(frozen=True)
class MotionSeries:
    """一个 trial 可选的八关节轨迹。

    ``t_s`` 为 ``[T]`` 秒，``joint_pos_rad`` 为 ``[T, 8]`` rad，速度若存在则为
    ``[T, 8]`` rad/s；三个数组已经使用 canonical 关节顺序和方向。
    """

    t_s: np.ndarray
    joint_pos_rad: np.ndarray
    joint_vel_rad_s: np.ndarray | None = None


@dataclass
class TrialRecord:
    """一个已发现 trial 的时间对齐数据与媒体文件索引。

    IMU、接触和运动数组已在数据集构造阶段加载；``rgb_paths``/``event_paths`` 只保存
    路径，等采样窗口时才解码。主要 shape：``t_s[T]``、``imu[T,6,9]``、三个 IMU
    分量 ``[T,6,3]``、``imu_valid[6]``、``contact_lr[T,2]``。
    """

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
        """返回同步后策略频率样本数 ``T``。"""

        return int(self.t_s.shape[0])


def _yaml_load(path: Path) -> Mapping[str, Any]:
    """读取 YAML 对象；文件不可读或顶层不是键值映射时给出明确错误。"""

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
    """把键值映射以 UTF-8 YAML 写入 ``path``，并自动创建父目录。"""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise DatasetFormatError("writing YAML mappings requires PyYAML") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(value), stream, allow_unicode=True, sort_keys=False)


def load_task_mapping(path: str | Path | None = None) -> dict[str, TaskSpec]:
    """加载并校验记录器使用的十个 BT/AT 任务编号。"""

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
    """加载六路 IMU 到机器人 link 的声明映射，此时尚不访问 USD Stage。"""

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
    """去除路径、大小写和标点差异，生成用于别名匹配的 USD body 名。"""

    token = str(name).replace("\\", "/").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]", "", token)


def resolve_imu_mapping(
    robot_link_names: Iterable[str],
    mapping: Mapping[str, Mapping[str, Any]] | str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, str | None]:
    """把配置中的六路 IMU link 与 USD articulation 实际 body 名对应起来。

    先比较规范化精确名称，再尝试 YAML 和内置别名；同一个 body 不会分配给两路 IMU。
    无法解析的项返回 ``None``。提供 ``output_path`` 时还会写出候选和 unresolved 列表，
    方便用户为非标准 link 名人工修订。

    Returns:
        六个 ``imu_XX`` 到真实 body 名或 ``None`` 的映射。
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
    """把数值秒、ISO 时间或 ``device+00T...`` 记录器时间解析为秒。"""

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
    # 同时处理 ISO 时间戳和记录仪的 ``device+00T00:10:00.005`` 格式。
    match = re.search(r"(?:T|^)(\d{1,3}):(\d{2}):(\d{2}(?:\.\d+)?)", text)
    if match:
        return float(match.group(1)) * 3600.0 + float(match.group(2)) * 60.0 + float(match.group(3))
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:us|µs)$", text.lower())
    if match:
        return float(match.group(1)) * 1.0e-6
    raise DatasetFormatError(f"cannot parse timestamp '{text}'")


def _column(row: Mapping[str, str], *names: str) -> float:
    """读取一行中首个非空别名列；缺失或无法转换时返回 ``NaN``。"""

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
    """读取一份 IMU CSV，并转换到仿真坐标系。

    函数识别多种历史列名，将 dps/degree 转成 SI 单位，按时间稳定排序、插补缺失单元，
    并去除重复时间戳。``sensor_index`` 是一基记录仪编号；提供后使用左右腿矩阵，
    省略则保留旧转换脚本使用的全局符号模式。

    Args:
        path: 单路 ``*_imu_XX.csv`` 文件。
        target_dt_s: CSV 无可用时间列时采用的样本间隔 [s]。
        calibration: 坐标标定；``None`` 使用默认标定。
        sensor_index: 可选的一基编号 1--6。

    Returns:
        包含 ``t_s[N]``、``orientation[N,3]`` [rad]、``gyro[N,3]`` [rad/s] 和
        ``accel[N,3]`` [m/s^2] 的 ``float32`` 数组字典（时间为 ``float64``）。

    Raises:
        DatasetFormatError: CSV 没有数据行或时间字段无法解析。
        OSError: 文件无法读取。
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
        """按候选列名前缀读取排序后的单通道 ``float64[N]`` 数组。"""

        # 前缀只能在字段边界匹配。像 ``gy`` 这样的短别名不能同时匹配
        # ``gyro_x_dps``/``gyro_y_dps``/``gyro_z_dps``，否则三个轴会悄悄读到第一列。
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
        # 若存在名称完全一致的 canonical 列，则优先使用它。
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
    # 记录仪列名为 *_dps，通用 gyro 列也按 dps 解释；只有显式的 *_rad_s/rads 后缀
    # 才表示数值已经是 SI 单位。
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
    # 插值/填补前先应用坐标系标定，使每个样本（包括为缺失单元合成的值）遵循同一约定。
    orientation, gyro, acc = calibration.apply_imu(orientation, gyro, acc, sensor_index=sensor_index)
    for array in (acc, gyro, orientation):
        # 单个非法单元不应毁掉整段 trial：保持最近的有限值；整条通道都缺失时才使用零。
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
    """把任意尾部 shape 的连续通道线性重采样到统一时间轴。

    Args:
        source_t: shape ``[S]`` 的源时间 [s]，应已排序且唯一。
        values: shape ``[S, ...]`` 的通道值。
        target_t: shape ``[T]`` 的目标时间 [s]。

    Returns:
        shape ``[T, ...]`` 的 ``float32`` 数组。全缺失通道填零，单点通道保持常值，
        超出源时间范围的部分保持首/末有效值。
    """

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
    """在 trial、数据根目录和绝对路径中解析媒体引用；找不到时返回 ``None``。"""

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
    """返回 trial 中首个含 ``task`` 或 ``files`` 字段的 JSON 元数据文件。"""

    for path in sorted(trial_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping) and ("task" in payload or "files" in payload):
            return path
    return None


def _discover_motion(trial_dir: Path, metadata_path: Path | None) -> Path | None:
    """按文件名关键字发现可选关节运动文件，并排除元数据、IMU 和压力记录。"""

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
    """从字段别名或嵌套对象提取 ``(时间, 关节角, 关节速度)`` 原始值。"""

    def pick(*names: str):
        """返回映射中首个存在的别名值；全部缺失时返回 ``None``。"""

        return next((payload[name] for name in names if name in payload), None)

    position = pick("joint_pos_rad", "joint_pos", "q_rad", "q", "joint_position")
    if position is None:
        nested = pick("motion", "target_motion", "kinematics")
        if isinstance(nested, Mapping):
            return _motion_mapping(nested)
        return None
    return (
        pick("t_s", "time_s", "time", "timestamps_s"),
        position,
        pick("joint_vel_rad_s", "joint_vel", "qdot_rad_s", "qdot"),
    )


def _read_motion(
    path: Path,
    target_dt_s: float,
    joint_sign: Sequence[float] = DEFAULT_JOINT_SIGN,
) -> MotionSeries | None:
    """读取可选关节轨迹，并应用 canonical 关节方向标定。

    支持 NPZ、PyTorch、JSON 与 CSV。位置是必需字段，速度缺失时使用时间梯度估算；
    文件内 ``joint_sign`` 可覆盖数据集级符号，适用于某次更换编码器后的记录。

    Args:
        path: 运动文件路径。
        target_dt_s: 缺少时间字段时采用的采样间隔 [s]。
        joint_sign: 八个关节的默认方向符号。

    Returns:
        校准后的 :class:`MotionSeries`；格式不受支持、没有位置字段或少于两个有效样本
        时返回 ``None``。

    Raises:
        DatasetFormatError: 存在运动数据但 shape、时间长度或符号配置不合法。
    """

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
    # motion 文件可以携带自己的标定元数据，并仅为当前 trial 覆盖数据集默认值；这样更换
    # 编码器后可以使用新标定，同时仍兼容旧记录。
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
    """读取压力 CSV，返回时间 [s] 与归一化平均压力；不可读或为空时返回 ``None``。"""

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
    """把事件文件解码为 ``[N,4] = (x,y,time_s,polarity)``。

    支持 NPZ/NPY、CSV、JSON 和可选 ``dv-processing`` 解码的 AEDAT4。时间戳会归一化
    为从零开始的秒，极性规范为 ``-1/+1``。预览视频会明确返回 ``None``，因为 RGB
    帧差不等于真实异步事件。

    Args:
        path: 左或右事件文件。

    Returns:
        ``float32[N,4]`` 数组；不支持的视频、缺少原生解码器或解码失败时返回
        ``None``。合法但安静的记录返回 shape ``[0,4]``，两者语义不同。

    Raises:
        DatasetFormatError: 已解码内容不是至少四列的二维数组。
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
                """按事件列别名读取浮点值，缺失或非法时返回 ``NaN``。"""

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
                        [[item["x"], item["y"], item["timestamp"], item["polarity"]] for item in array]
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
    # AEDAT 时间戳通常使用微秒，这里统一换算为秒。
    magnitude = float(np.nanmax(np.abs(array[:, 2])))
    if magnitude > 1.0e5:
        array[:, 2] *= 1.0e-6
    array[:, 2] -= array[:, 2].min()
    array[:, 3] = np.where(array[:, 3] > 0, 1.0, -1.0)
    return array.astype(np.float32)


class ExoskeletonImitationDataset(Dataset):
    """从外骨骼实测 trial 中提供时间同步窗口的 PyTorch 数据集。

    类本身不依赖 Isaac Lab。``task_id`` 可为单个编号、编号序列或全部任务。随机模式
    使用 ``seed + epoch + index/抽样计数`` 构造可复现随机流；顺序模式则遍历每个
    trial 的所有合法起点。缺少可选媒体或解码器时仍返回固定 shape 零张量，同时把
    对应 ``valid`` 设为 ``False``，保证 PyTorch 批处理不会因键或 shape 变化而失败。

    ``calibration_path`` 决定 IMU 轴和八关节方向映射；``None`` 自动使用仓库默认值。
    非媒体数据在构造时读取，RGB 和 Event 在 :meth:`_sample_record` 中延迟解码。

    返回的事件流 ``target_event_stream[N,4]`` 四列为
    ``(x, y, t_relative_s, polarity)``；``target_event_frame_index[N]`` 把
    每个事件映射到策略频率窗口，``target_event`` 则是 PPO 可批处理的
    ``[window,2 cameras,2 polarities,H,W]`` 计数直方图。``event_source_size`` 使用
    ``(height,width)``，默认对应数据中的 240x320 DVXplorer。
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
        """发现、校验并索引数据集中的 trial。

        Args:
            root: 包含 ``Trial*`` 子目录的数据集根目录。
            task_id: 一个或多个 BT/AT 编号；``None``/``all``/``none`` 表示全部。
            window_size: 每个样本包含的策略时间步数。
            target_dt_s: 统一后的策略采样间隔 [s]，默认 ``0.02`` 即 50 Hz。
            random_window: 是否随机选择 trial 内的窗口起点。
            random_task_sampling: 未指定任务时是否先均匀抽任务，再抽 trial。
            steps_per_epoch: 可选的 ``__len__`` 固定值，便于控制训练 epoch 长度。
            seed: 随机窗口的基础种子。
            rgb_size: RGB 输出 ``(height,width)`` [pixel]。
            event_size: 事件直方图输出 ``(height,width)`` [bin]。
            event_source_size: 原始事件传感器 ``(height,width)`` [pixel]；``None`` 从
                当前事件坐标范围保守推断。
            load_rgb: 是否在采样窗口时延迟读取 RGB。
            load_event: 是否在采样窗口时延迟解码事件。
            strict: ``True`` 时遇到坏 trial 立即报错；否则记录警告并跳过。
            task_mapping_path: 可选任务名称 YAML。
            calibration_path: 可选坐标/方向标定 YAML。

        Raises:
            ValueError: 时间间隔、窗口或图像尺寸不合法，或任务编号未知。
            FileNotFoundError: 数据集根目录或显式标定文件不存在。
            DatasetFormatError: 没有可用 trial，或严格模式下发现格式错误。
        """

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
        self.event_source_size = None if event_source_size is None else tuple(int(value) for value in event_source_size)
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
        # ``sample()`` 由环境 reset 直接调用，而不是经 PyTorch DataLoader 调用，因此需要
        # 内部计数器，避免每次 reset 都抽到完全相同的 seeded 窗口。
        self._sample_counter = 0

    def _normalise_task_ids(self, task_id: str | Sequence[str] | None) -> tuple[str, ...]:
        """把单个、多个或 ``all`` 任务选择规范为已校验的大写任务编号元组。"""

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
        """递归发现 ``Trial*`` 目录并加载可用的非媒体通道。

        Returns:
            按目录名排序的 :class:`TrialRecord` 列表。

        Raises:
            DatasetFormatError: 根路径不是目录，或严格模式下无法判断任务编号。
            OSError: 严格模式下底层文件读取失败。
        """

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
        """把一个 trial 的六路 IMU、压力和运动轨迹对齐到同一时间轴。

        六路 IMU 先各自解析，再取共同时间区间并以 ``target_dt_s`` 重采样。缺失某一路
        时填 ``[T,3]`` 零值并将对应 ``imu_valid`` 设为 ``False``。RGB/Event 此时只
        记录路径；关节运动若存在则立即解析和对齐。

        Args:
            trial_dir: 当前 ``TrialXX`` 目录。
            task_id: 已确认的 BT/AT 任务编号。
            subject_id: 元数据或目录推断出的受试者编号。
            metadata_path: 可选元数据 JSON 路径。
            payload: 已读取的元数据映射。

        Returns:
            时间从零开始、非媒体通道已同步的 :class:`TrialRecord`。

        Raises:
            DatasetFormatError: 六路腿部 IMU 全部缺失或必需数据无法解释。
            OSError: 必需文件无法读取。
        """

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
                    contact[:, side] = _resample_channels(pressure_t, pressure_values[:, None], target_absolute)[
                        :, 0
                    ].clip(0.0, 1.0)

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
            # 元数据指向 *_event_preview.mp4 时，优先寻找原始 AEDAT4 文件。
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
        """把关节角和可选关节速度线性重采样到目标时间 ``target_t`` [s]。"""

        position = _resample_channels(motion.t_s, motion.joint_pos_rad, target_t)
        velocity = (
            None if motion.joint_vel_rad_s is None else _resample_channels(motion.t_s, motion.joint_vel_rad_s, target_t)
        )
        return MotionSeries(target_t.astype(np.float32), position, velocity)

    def set_epoch(self, epoch: int) -> None:
        """切换随机窗口的确定性随机流，并把环境抽样计数归零。"""

        self.epoch = int(epoch)
        self._sample_counter = 0

    @property
    def task_ids(self) -> tuple[str, ...]:
        """返回已发现 trial 实际包含的排序后任务编号。"""

        return tuple(sorted({record.task_id for record in self.trials}))

    def __len__(self) -> int:
        """返回当前 epoch 可索引的窗口数；随机模式至少返回一个样本。"""

        if self.steps_per_epoch is not None:
            return max(1, int(self.steps_per_epoch))
        if self.random_window or self.random_task_sampling:
            return max(1, len(self.trials))
        return sum(max(1, record.length - self.window_size + 1) for record in self.trials)

    def _index_to_trial_start(self, index: int) -> tuple[int, int]:
        """把 PyTorch 样本索引映射为 ``(trial索引, 窗口起点)``。

        随机模式仍由 ``seed``、``epoch`` 和 ``index`` 唯一决定，所以同一输入可复现；
        顺序模式会把所有 trial 的窗口数视作一段连续索引空间。

        Args:
            index: ``Dataset.__getitem__`` 收到的整数索引。

        Returns:
            ``(trial_index, start_sample)``，两项均为零基整数。
        """

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
        """把整数索引解析为 trial 窗口，并返回一份训练样本字典。"""

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
        """按任务、trial 或时间戳为环境抽取一个窗口。

        这个接口主要由环境 reset 调用，不经过 ``DataLoader.__getitem__``，因此使用
        内部递增计数让同一 epoch 中连续 reset 不会永远拿到同一随机窗口。

        Args:
            task_id: 可选 BT/AT 编号；``None`` 允许从全部已加载任务中选择。
            timestamp_s: 可选窗口起始时间 [s]；提供后选择不早于该时刻的样本。
            window_size: 临时覆盖默认窗口长度。
            trial_id: 可选 ``TrialXX`` 名称过滤。
            random_window: 临时覆盖随机起点设置。

        Returns:
            包含元数据、同步目标张量和各模态有效掩码的字典。

        Raises:
            KeyError: 没有 trial 同时满足任务和 trial 过滤条件。
            ValueError: ``window_size`` 不是正数。
        """

        requested = None if task_id is None else str(task_id).upper()
        candidates = [record for record in self.trials if requested is None or record.task_id == requested]
        if trial_id is not None:
            candidates = [record for record in candidates if record.trial_id == trial_id]
        if not candidates:
            raise KeyError(f"no trial matches task_id={task_id!r}, trial_id={trial_id!r}")
        # 把单调递增的抽样序号加入随机种子：固定 seed/epoch 时仍可复现，同时确保各次独立
        # reset 不会重复使用同一个冻结目标窗口。
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
        """按确定时间戳采样的便捷别名，强制关闭随机窗口。"""

        return self.sample(task_id, timestamp_s, window_size=window_size, random_window=False)

    def _sample_record(self, record: TrialRecord, start: int, size: int | None = None) -> dict[str, Any]:
        """从单个 trial 组装完整训练样本，并按需解码媒体。

        超过 trial 末尾的窗口位置重复最后一帧，从而始终返回请求的 ``size``。运动
        文件缺失时 ``target_motion/velocity`` 为零且 ``motion_valid=False``。

        Args:
            record: 已加载非媒体通道的 trial。
            start: 零基窗口起始样本。
            size: 可选窗口长度；``None`` 使用数据集默认值。

        Returns:
            主要 shape 为：时间 ``[W]``、运动 ``[W,8]``、IMU ``[W,6,9]``、接触
            ``[W,2]``、RGB ``[W,2,3,H,W]``、事件直方图 ``[W,2,2,H,W]``；事件原始
            流长度记为 ``E``，对应 ``[E,4]`` 及两个 ``[E]`` 索引。

        Raises:
            ValueError: 有效窗口长度不是正数。
        """

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
        """按窗口时间延迟读取左右 RGB 帧。

        Args:
            record: 提供左右 RGB 文件路径的 trial。
            t_window: shape ``[W]`` 的目标时间 [s]。

        Returns:
            ``(frames, valid)``：frames 是归一化到 ``[0,1]`` 的
            ``float32[W,2,3,H,W]``，valid 是左右相机 ``bool[2]``。一路中任一目标帧
            解码失败时，该路保持全零并标为无效。
        """

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
        """读取并缩放某一时刻的一张 RGB 帧。

        NPY/NPZ 直接按策略时间索引；视频优先复用 imageio reader，失败后复用 OpenCV
        capture。两种解码器都不可用或帧格式非法时返回 ``None``，不会抛出并中断整个
        trial。

        Args:
            path: NPY/NPZ 图像数组或视频文件。
            time_s: 相对 trial 起点的目标时间 [s]。

        Returns:
            ``float32[3,H,W]``、值域 ``[0,1]`` 的 RGB 张量；失败时返回 ``None``。
        """

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
                except Exception as imageio_exc:  # 可选 ffmpeg/PyAV 解码器
                    # OpenCV 是常见的 Isaac/ROS 依赖；即使缺少 imageio 的 ffmpeg 插件，
                    # 也能解码提供的 MP4。它只作为视频解码后备，事件文件不会进入此路径。
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
        """延迟解码并截取窗口内左右事件，生成固定直方图。

        Args:
            record: 含左右事件文件路径的 trial。
            start_s: 窗口起点 [s]。
            end_s: 窗口右开区间终点 [s]。
            window_size: 策略时间 bin 数。

        Returns:
            五元组 ``(histogram, stream, frame_index, camera_index, valid)``。shape 分别为
            ``[W,2,2,H,W]``、``[E,4]``、``[E]``、``[E]`` 和布尔值。成功解码但本
            窗口事件数为零时 ``valid`` 仍为 ``True``；缺失/损坏/不支持才为 ``False``。
        """

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
        # DVS 记录只要成功解码，即使短窗口内没有阈值越过也仍然有效。零事件区间表示可观测
        # 的安静场景；``False`` 只表示事件流缺失、不受支持或损坏（已在上方处理）。
        return histogram, stream, event_frame_index, event_camera_index, True

    def _event_grid_coordinates(
        self, x: np.ndarray, y: np.ndarray, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """把原始传感器像素缩放到 PPO 固定事件网格。

        数据中的 DVXplorer 是 320x240，而目标通常是 64x64；直接裁剪会让大部分事件
        堆到最后一行/列，因此这里按源分辨率缩放。源尺寸为 ``None`` 时才根据当前
        坐标范围推断；默认固定尺寸可避免短窗口没覆盖边缘像素造成比例漂移。

        Args:
            x: 原始水平像素坐标 ``[E]``。
            y: 原始垂直像素坐标 ``[E]``。
            width: 目标网格宽度。
            height: 目标网格高度。

        Returns:
            两个 ``int64[E]`` 数组，分别是裁剪到 ``[0,width-1]`` 和
            ``[0,height-1]`` 的网格坐标。
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
        """释放延迟解码期间缓存的 imageio/OpenCV 视频句柄。

        Returns:
            无返回值。重复调用安全，单个解码器关闭失败不会阻止其他句柄释放。
        """

        for reader in self._video_readers.values():
            with contextlib.suppress(Exception):
                reader.close()
        self._video_readers.clear()
        for capture in self._video_captures.values():
            with contextlib.suppress(Exception):
                capture.release()
        self._video_captures.clear()

    def __del__(self):  # pragma: no cover - interpreter shutdown order is undefined
        """对象回收时尽力关闭视频句柄；解释器退出阶段的异常会被忽略。"""

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
