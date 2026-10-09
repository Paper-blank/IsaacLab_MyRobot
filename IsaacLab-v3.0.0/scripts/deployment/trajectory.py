# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""读取、校验并插值离线外骨骼轨迹。

本模块位于部署数据流的起点，由 :mod:`scripts.deploy_prediction` 读取
``prediction.json`` 或项目生成的 ``ruleresult/v1`` JSONL。输入文件中的关节顺序、
时间戳和数值经过校验后，被统一为按时间递增的 :class:`Trajectory`；
:func:`deployment.replay.replay` 再按每个硬件控制时刻调用 :meth:`Trajectory.sample`
得到八关节目标。

所有返回值都使用普通 Python 类型，不依赖 Isaac Lab、PyTorch 或 NumPy，因此同一
加载器可以复制到资源较少的工控机上。建议初学者按“``JointSample`` 单个采样 ->
``Trajectory`` 整条轨迹 -> ``load_trajectory`` 文件入口”的顺序阅读，最后再看
Hermite 三次插值和五次插值公式。
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Isaac Lab 可用时直接复用唯一的项目关节顺序定义。
    from isaaclab_assets.tasks.robot_contract import ACTUATED_JOINT_NAMES
except Exception:  # pragma: no cover - 裸工控机没有 Isaac Lab 时使用同一顺序的副本。
    ACTUATED_JOINT_NAMES = (
        "base_to_Rturn",
        "Rturn_to_Rthigh",
        "Rthigh_to_Rleg",
        "Rleg_to_Rfoot",
        "base_to_Lturn",
        "Lturn_to_Lthigh",
        "Lthigh_to_Lleg",
        "Lleg_to_Lfoot",
    )


JOINT_COUNT = len(ACTUATED_JOINT_NAMES)


class TrajectoryError(ValueError):
    """轨迹格式、关节顺序、时间戳或数值不适合安全部署时抛出的异常。"""


def _finite(value: Any, label: str) -> float:
    """把一个输入值转换为有限浮点数。

    Args:
        value: JSON 中读到的数值或可转换为浮点数的值。
        label: 出错时用于定位字段的名称。

    Returns:
        转换后的有限 ``float``。

    Raises:
        TrajectoryError: 输入不是数值，或为 NaN/正负无穷大。
    """

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrajectoryError(f"{label} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise TrajectoryError(f"{label} must be finite, got {value!r}")
    return number


def _vector(value: Any, label: str, *, names: Iterable[str] | None = None) -> tuple[float, ...]:
    """校验一个八关节向量，并按项目规范关节顺序重排。

    Args:
        value: 长度为八的一维列表或元组，shape 为 ``[8]``。
        label: 出错时用于定位 JSON 字段的名称。
        names: ``value`` 当前使用的关节名顺序。传入时必须恰好包含全部八个规范
            关节；不传时表示 ``value`` 已经采用规范顺序。

    Returns:
        全部元素转换为有限 ``float``、并按照 ``ACTUATED_JOINT_NAMES`` 排列的元组，
        shape 为 ``[8]``。

    Raises:
        TrajectoryError: 向量长度错误、含非有限值，或关节名有缺失/多余项。
    """

    if not isinstance(value, (list, tuple)) or len(value) != JOINT_COUNT:
        raise TrajectoryError(f"{label} must contain {JOINT_COUNT} values")
    values = tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))
    if names is None:
        return values
    source_names = tuple(str(name) for name in names)
    if len(source_names) != JOINT_COUNT or set(source_names) != set(ACTUATED_JOINT_NAMES):
        missing = sorted(set(ACTUATED_JOINT_NAMES) - set(source_names))
        extra = sorted(set(source_names) - set(ACTUATED_JOINT_NAMES))
        raise TrajectoryError(f"joint_names mismatch; missing={missing}, extra={extra}")
    return tuple(values[source_names.index(name)] for name in ACTUATED_JOINT_NAMES)


def _field(row: dict[str, Any], *names: str) -> Any:
    """按给定优先级读取第一个存在的兼容字段名。

    Args:
        row: 一条 JSON 记录。
        *names: 从新到旧排列的候选字段名。

    Returns:
        第一个已存在字段的原始值；全部不存在时返回 ``None``。
    """

    for name in names:
        if name in row:
            return row[name]
    return None


@dataclass(frozen=True)
class JointSample:
    """轨迹中某一时刻的一组完整八关节目标。

    Attributes:
        timestamp_s: 采样时间 [s]。经过 :func:`load_trajectory` 默认归一化后，
            它是相对轨迹起点的时间。
        angle_rad: 八个规范关节的目标角度 [rad]，shape 为 ``[8]``。
        velocity_rad_s: 八个规范关节的目标角速度 [rad/s]，shape 为 ``[8]``。
        acceleration_rad_s2: 八个规范关节的目标角加速度 [rad/s^2]，
            shape 为 ``[8]``。

    三个向量在本模块中始终使用 ``ACTUATED_JOINT_NAMES`` 顺序。映射到实体电机顺序
    和电机侧数值是 :mod:`deployment.joint_mapping` 的职责。
    """

    timestamp_s: float
    angle_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    acceleration_rad_s2: tuple[float, ...]

    def __post_init__(self) -> None:
        """把数值固定为浮点元组，并拒绝负时间或非法八关节向量。"""

        timestamp = _finite(self.timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise TrajectoryError("timestamp_s must be non-negative after normalization")
        object.__setattr__(self, "timestamp_s", timestamp)
        for field_name in ("angle_rad", "velocity_rad_s", "acceleration_rad_s2"):
            values = getattr(self, field_name)
            if len(values) != JOINT_COUNT or not all(math.isfinite(float(item)) for item in values):
                raise TrajectoryError(f"{field_name} must contain {JOINT_COUNT} finite values")
            object.__setattr__(self, field_name, tuple(float(item) for item in values))


def _lerp(first: tuple[float, ...], second: tuple[float, ...], fraction: float) -> tuple[float, ...]:
    """对两个同形向量逐元素做线性插值。

    Args:
        first: 区间左端向量，shape 为 ``[8]``。
        second: 区间右端向量，shape 为 ``[8]``。
        fraction: 查询点在区间内的归一化位置，通常位于 ``[0, 1]``。

    Returns:
        ``first + (second - first) * fraction``，shape 为 ``[8]``。
    """

    return tuple(a + (b - a) * fraction for a, b in zip(first, second))


def _hermite3(left: JointSample, right: JointSample, time_s: float) -> JointSample:
    """用三次 Hermite 多项式在两个采样之间插值位置、速度和加速度。

    Hermite 插值同时满足左右端点的位置与速度，因此位置和速度连接较平滑；返回的
    加速度由同一个位置多项式二阶求导得到。它不使用文件中的端点加速度。

    Args:
        left: 查询时刻左侧的八关节采样。
        right: 查询时刻右侧的八关节采样，时间必须晚于 ``left``。
        time_s: 位于左右端点之间的查询时间 [s]。

    Returns:
        查询时刻的位置 [rad]、速度 [rad/s] 和加速度 [rad/s^2]，各 shape ``[8]``。
    """

    dt = right.timestamp_s - left.timestamp_s
    u = (time_s - left.timestamp_s) / dt
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    dh00 = 6.0 * u2 - 6.0 * u
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = -6.0 * u2 + 6.0 * u
    dh11 = 3.0 * u2 - 2.0 * u
    d2h00 = 12.0 * u - 6.0
    d2h10 = 6.0 * u - 4.0
    d2h01 = -12.0 * u + 6.0
    d2h11 = 6.0 * u - 2.0
    angle = tuple(
        h00 * q0 + h10 * dt * v0 + h01 * q1 + h11 * dt * v1
        for q0, v0, q1, v1 in zip(left.angle_rad, left.velocity_rad_s, right.angle_rad, right.velocity_rad_s)
    )
    velocity = tuple(
        (dh00 * q0 + dh10 * dt * v0 + dh01 * q1 + dh11 * dt * v1) / dt
        for q0, v0, q1, v1 in zip(left.angle_rad, left.velocity_rad_s, right.angle_rad, right.velocity_rad_s)
    )
    acceleration = tuple(
        (d2h00 * q0 + d2h10 * dt * v0 + d2h01 * q1 + d2h11 * dt * v1) / (dt * dt)
        for q0, v0, q1, v1 in zip(left.angle_rad, left.velocity_rad_s, right.angle_rad, right.velocity_rad_s)
    )
    return JointSample(time_s, angle, velocity, acceleration)


def _quintic(left: JointSample, right: JointSample, time_s: float) -> JointSample:
    """用五次多项式在两个采样之间连续匹配位置、速度和加速度。

    对每个关节独立求六个多项式系数，使左右端点的角度、角速度和角加速度都与
    输入一致。与线性插值相比它更平滑，但前提是文件中的速度和加速度可信。

    Args:
        left: 查询时刻左侧的八关节采样。
        right: 查询时刻右侧的八关节采样，时间必须晚于 ``left``。
        time_s: 位于左右端点之间的查询时间 [s]。

    Returns:
        查询时刻的位置 [rad]、速度 [rad/s] 和加速度 [rad/s^2]，各 shape ``[8]``。
    """

    dt = right.timestamp_s - left.timestamp_s
    u = (time_s - left.timestamp_s) / dt
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    outputs: list[tuple[float, float, float]] = []
    for q0, v0, a0, q1, v1, a1 in zip(
        left.angle_rad,
        left.velocity_rad_s,
        left.acceleration_rad_s2,
        right.angle_rad,
        right.velocity_rad_s,
        right.acceleration_rad_s2,
    ):
        c0 = q0
        c1 = dt * v0
        c2 = 0.5 * dt * dt * a0
        aa = q1 - (c0 + c1 + c2)
        bb = dt * v1 - (c1 + 2.0 * c2)
        cc = dt * dt * a1 - 2.0 * c2
        c3 = 10.0 * aa - 4.0 * bb + 0.5 * cc
        c4 = -15.0 * aa + 7.0 * bb - cc
        c5 = 6.0 * aa - 3.0 * bb + 0.5 * cc
        q = c0 + c1 * u + c2 * u2 + c3 * u3 + c4 * u4 + c5 * u5
        v = (c1 + 2.0 * c2 * u + 3.0 * c3 * u2 + 4.0 * c4 * u3 + 5.0 * c5 * u4) / dt
        a = (2.0 * c2 + 6.0 * c3 * u + 12.0 * c4 * u2 + 20.0 * c5 * u3) / (dt * dt)
        outputs.append((q, v, a))
    return JointSample(
        time_s,
        tuple(item[0] for item in outputs),
        tuple(item[1] for item in outputs),
        tuple(item[2] for item in outputs),
    )


@dataclass(frozen=True)
class Trajectory:
    """经过校验、严格按时间递增且支持插值的八关节轨迹。

    Attributes:
        samples: 一个或多个 :class:`JointSample`，时间戳必须严格递增。
        joint_names: 固定的八关节规范顺序，shape 为 ``[8]``。
    """

    samples: tuple[JointSample, ...]
    joint_names: tuple[str, ...] = tuple(ACTUATED_JOINT_NAMES)

    def __post_init__(self) -> None:
        """冻结输入序列，并检查非空、关节顺序和严格递增时间戳。"""

        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        if not self.samples:
            raise TrajectoryError("trajectory contains no samples")
        if tuple(self.joint_names) != tuple(ACTUATED_JOINT_NAMES):
            raise TrajectoryError("trajectory joint order must match ACTUATED_JOINT_NAMES")
        previous = -math.inf
        for sample in self.samples:
            if sample.timestamp_s <= previous:
                raise TrajectoryError("trajectory timestamps must be strictly increasing")
            previous = sample.timestamp_s

    @property
    def duration_s(self) -> float:
        """返回归一化轨迹终点时间 [s]。"""

        return self.samples[-1].timestamp_s

    @property
    def points(self) -> tuple[JointSample, ...]:
        """返回全部采样；这是为旧调用方保留的 ``samples`` 别名。"""

        return self.samples

    @property
    def joint_count(self) -> int:
        """返回规范关节数量，当前固定为八。"""

        return JOINT_COUNT

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_joint_names: Sequence[str] = ACTUATED_JOINT_NAMES,
        episode: int | None = None,
        use_nested_prediction: bool = False,
    ) -> Trajectory:
        """通过 :func:`load_trajectory` 从 JSON/JSONL 创建轨迹。

        Args:
            path: 输入 JSON 或 JSONL 文件路径。
            expected_joint_names: 调用方期望的关节顺序；必须等于项目规范顺序。
            episode: 多 episode JSONL 中要选择的 episode 编号。
            use_nested_prediction: 是否读取记录内的未来预测序列。

        Returns:
            完成解析、排序、去重和默认起点归一化的轨迹。

        Raises:
            TrajectoryError: 期望关节顺序错误或文件内容不合法。
        """

        if tuple(expected_joint_names) != tuple(ACTUATED_JOINT_NAMES):
            raise TrajectoryError("expected_joint_names must use the canonical joint order")
        return load_trajectory(
            path,
            episode=episode,
            use_nested_prediction=use_nested_prediction,
        )

    def sample(self, time_s: float, method: str = "linear") -> JointSample:
        """在任意控制时刻查询轨迹，必要时执行插值。

        Args:
            time_s: 相对轨迹起点的查询时间 [s]。
            method: ``linear``（逐字段线性）、``hermite3``（位置/速度连续的三次
                Hermite）或 ``quintic``（位置/速度/加速度连续的五次多项式）。

        Returns:
            查询时刻的八关节采样。查询早于起点或晚于终点时保持最近端点，不进行
            外推，避免在文件范围之外生成未经验证的动作。

        Raises:
            TrajectoryError: 查询时间不是有限值，或插值方法未知。
        """


        query = _finite(time_s, "sample time")
        method = method.lower()
        if method not in {"linear", "hermite3", "quintic"}:
            raise TrajectoryError(f"unknown interpolation method {method!r}")
        if query <= self.samples[0].timestamp_s:
            return self.samples[0]
        if query >= self.samples[-1].timestamp_s:
            return self.samples[-1]
        timestamps = [sample.timestamp_s for sample in self.samples]
        right_index = bisect.bisect_right(timestamps, query)
        left = self.samples[right_index - 1]
        right = self.samples[right_index]
        fraction = (query - left.timestamp_s) / (right.timestamp_s - left.timestamp_s)
        if method == "linear":
            return JointSample(
                query,
                _lerp(left.angle_rad, right.angle_rad, fraction),
                _lerp(left.velocity_rad_s, right.velocity_rad_s, fraction),
                _lerp(left.acceleration_rad_s2, right.acceleration_rad_s2, fraction),
            )
        if method == "hermite3":
            return _hermite3(left, right, query)
        return _quintic(left, right, query)


def _read_json_documents(path: Path) -> list[Any]:
    """读取单个 JSON 文档或逐行 JSON（JSONL）文件。

    函数先尝试把整个文件解析为一个 JSON 文档；失败后再把每个非空行分别解析。
    因此普通 JSON 数组、普通 JSON 对象和项目追加写入的 JSONL 都能进入同一流程。

    Args:
        path: UTF-8 编码的轨迹文件路径。

    Returns:
        顶层 JSON 文档列表；单个 JSON 文件的列表长度为一。

    Raises:
        TrajectoryError: 文件不可读、为空，或某一 JSONL 行语法错误。
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrajectoryError(f"cannot read trajectory {path}: {exc}") from exc
    if not text.strip():
        raise TrajectoryError(f"trajectory file {path} is empty")
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        documents: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TrajectoryError(f"invalid JSON on line {line_number}: {exc}") from exc
        if not documents:
            raise TrajectoryError(f"trajectory file {path} has no JSON records")
        return documents


def _records(documents: list[Any]) -> list[dict[str, Any]]:
    """把多种顶层 JSON 包装形式展平为逐时刻记录列表。

    对包含 ``trajectory``、``waypoints`` 或 ``records`` 数组的对象，顶层
    ``schema``、``joint_names`` 和 ``episode`` 会继承到数组中的每条记录；记录
    自己的同名字段优先。

    Args:
        documents: :func:`_read_json_documents` 返回的 JSON 文档列表。

    Returns:
        展平并继承公共元数据后的记录字典列表。

    Raises:
        TrajectoryError: 顶层或数组元素不是受支持的 JSON 对象形式。
    """

    result: list[dict[str, Any]] = []
    for document in documents:
        if isinstance(document, list):
            values = document
            inherited: dict[str, Any] = {}
        elif isinstance(document, dict):
            values = _field(document, "trajectory", "waypoints", "records")
            if isinstance(values, list):
                inherited = {
                    key: document[key]
                    for key in ("schema", "joint_names", "episode")
                    if key in document
                }
            else:
                values = [document]
                inherited = {}
        else:
            raise TrajectoryError("top-level JSON must be an object or list")
        for value in values:
            if not isinstance(value, dict):
                raise TrajectoryError("each trajectory record must be a JSON object")
            merged = dict(inherited)
            merged.update(value)
            result.append(merged)
    return result


def _expand_record(
    row: dict[str, Any],
    *,
    use_nested_prediction: bool,
    default_dt_s: float,
) -> list[JointSample]:
    """把一条 JSON 记录展开为一个或多个规范八关节采样。

    支持三种输入：普通 ``[8]`` 单点、一条记录中的 ``[H, 8]`` 序列，以及启用
    ``use_nested_prediction`` 后的预测窗口。其中 ``H`` 是该记录包含的未来步数。

    Args:
        row: 已继承公共元数据的一条 JSON 记录。
        use_nested_prediction: 是否优先展开 ``prediction`` 内的未来窗口。
        default_dt_s: 序列缺少逐点时间戳时使用的采样间隔 [s]。

    Returns:
        一个或多个使用规范关节顺序的 :class:`JointSample`。

    Raises:
        TrajectoryError: schema、预测状态、shape、有效掩码、关节顺序或时间戳非法。
    """

    schema = row.get("schema")
    if schema is not None and str(schema) != "ruleresult/v1":
        raise TrajectoryError(f"unsupported trajectory schema {schema!r}")
    names = row.get("joint_names")
    timestamp_value = _field(row, "timestamp_s", "timestamp", "time_s")
    timestamp = 0.0 if timestamp_value is None else _finite(timestamp_value, "timestamp_s")
    if timestamp < 0.0:
        raise TrajectoryError("timestamps must be non-negative")

    if use_nested_prediction and isinstance(row.get("prediction"), dict):
        prediction = row["prediction"]
        status = str(row.get("prediction_status", "available")).lower()
        if status not in {"available", "ok", "ready"}:
            raise TrajectoryError(f"prediction_status={status!r} is not deployable")
        angles = _field(prediction, "angle_rad", "angles", "angle_targets_rad")
        velocities = _field(prediction, "velocity_rad_s", "velocities")
        accelerations = _field(prediction, "angular_acceleration_rad_s2", "accelerations")
        times = prediction.get("timestamps_s")
        if not isinstance(angles, list) or not angles or not isinstance(times, list):
            raise TrajectoryError("prediction must contain timestamps_s and non-empty angle_rad sequence")
        if len(times) != len(angles):
            raise TrajectoryError("prediction timestamps and angle sequence lengths differ")
        if velocities is None:
            velocities = [[0.0] * JOINT_COUNT for _ in angles]
        if accelerations is None:
            accelerations = [[0.0] * JOINT_COUNT for _ in angles]
        if len(velocities) != len(angles) or len(accelerations) != len(angles):
            raise TrajectoryError("prediction angle, velocity, and acceleration lengths differ")
        valid_mask = prediction.get("valid_mask")
        if valid_mask is not None:
            if not isinstance(valid_mask, list) or len(valid_mask) != len(angles):
                raise TrajectoryError("prediction valid_mask length differs from the horizon")
            if not all(isinstance(valid, bool) for valid in valid_mask):
                raise TrajectoryError("prediction valid_mask must contain booleans")
            selected = [index for index, valid in enumerate(valid_mask) if bool(valid)]
            if not selected:
                raise TrajectoryError("prediction valid_mask contains no usable samples")
            angles = [angles[index] for index in selected]
            velocities = [velocities[index] for index in selected]
            accelerations = [accelerations[index] for index in selected]
            times = [times[index] for index in selected]
        numeric_times = [_finite(value, f"prediction.timestamps_s[{index}]") for index, value in enumerate(times)]
        if any(value <= 0.0 for value in numeric_times):
            raise TrajectoryError("prediction timestamps must be positive relative times")
        if any(right <= left for left, right in zip(numeric_times, numeric_times[1:])):
            raise TrajectoryError("prediction timestamps must be strictly increasing")
        samples: list[JointSample] = []
        for index, relative in enumerate(numeric_times):
            samples.append(
                JointSample(
                    timestamp + relative,
                    _vector(angles[index], "prediction.angle_rad", names=names),
                    _vector(velocities[index], "prediction.velocity_rad_s", names=names),
                    _vector(accelerations[index], "prediction.angular_acceleration_rad_s2", names=names),
                )
            )
        return samples

    angles = _field(row, "angle_rad", "angles", "angle_targets_rad")
    velocities = _field(row, "velocity_rad_s", "velocities")
    accelerations = _field(row, "angular_acceleration_rad_s2", "accelerations")
    if angles is None:
        raise TrajectoryError("record is missing angle_rad (or angles)")
    # 除常见的单点 [8] 外，也允许一条记录直接携带 H 个未来点 [H, 8]。
    if isinstance(angles, list) and angles and isinstance(angles[0], (list, tuple)):
        angle_rows = angles
        velocity_rows = velocities if isinstance(velocities, list) else [[0.0] * JOINT_COUNT for _ in angle_rows]
        acceleration_rows = (
            accelerations if isinstance(accelerations, list) else [[0.0] * JOINT_COUNT for _ in angle_rows]
        )
        times = row.get("timestamps_s")
        if not isinstance(times, list):
            times = [timestamp + index * default_dt_s for index in range(len(angle_rows))]
        if (
            len(times) != len(angle_rows)
            or len(velocity_rows) != len(angle_rows)
            or len(acceleration_rows) != len(angle_rows)
        ):
            raise TrajectoryError("sequence record fields have different lengths")
        return [
            JointSample(
                _finite(times[index], f"timestamps_s[{index}]"),
                _vector(angle_rows[index], "angle_rad", names=names),
                _vector(velocity_rows[index], "velocity_rad_s", names=names),
                _vector(acceleration_rows[index], "angular_acceleration_rad_s2", names=names),
            )
            for index in range(len(angle_rows))
        ]
    return [
        JointSample(
            timestamp,
            _vector(angles, "angle_rad", names=names),
            _vector(velocities if velocities is not None else [0.0] * JOINT_COUNT, "velocity_rad_s", names=names),
            _vector(
                accelerations if accelerations is not None else [0.0] * JOINT_COUNT,
                "angular_acceleration_rad_s2",
                names=names,
            ),
        )
    ]


def load_trajectory(
    path: str | Path,
    *,
    episode: int | None = None,
    use_nested_prediction: bool = False,
    default_dt_s: float = 0.02,
    normalize_start: bool = True,
) -> Trajectory:
    """读取、校验、排序并去重一个 JSON/JSONL 轨迹文件。

    嵌套预测必须显式启用，因为 JSONL 中相邻记录的预测窗口会重叠，它们是每一时刻
    对未来的预测，并不天然构成一条连续轨迹。启用后，相对预测时间会转换为
    ``顶层 timestamp_s + prediction 内相对 timestamp_s``；窗口在同一时刻重叠时
    保留文件中后出现的值。若文件包含多个 episode，调用方必须明确选择一个，避免
    把追加文件中的旧 episode 一起回放。

    Args:
        path: 输入 JSON 或 JSONL 文件路径。
        episode: 要选择的 episode 整数；单 episode 文件可不传。
        use_nested_prediction: 是否使用每条记录内的 ``prediction`` 未来窗口。
        default_dt_s: 普通记录缺少时间戳时采用的相邻采样间隔 [s]，默认 0.02 s。
        normalize_start: 是否把第一个有效采样平移到 ``t=0``。

    Returns:
        使用规范关节顺序、严格按时间递增的 :class:`Trajectory`。

    Raises:
        TrajectoryError: 文件、episode、时间戳、数值、向量 shape 或关节顺序无效。
    """

    default_dt_s = _finite(default_dt_s, "default_dt_s")
    if default_dt_s <= 0.0:
        raise TrajectoryError("default_dt_s must be positive")
    source = Path(path).expanduser()
    rows = _records(_read_json_documents(source))
    for row in rows:
        episode_value = row.get("episode")
        if episode_value is not None and (
            isinstance(episode_value, bool) or not isinstance(episode_value, (int, str))
        ):
            raise TrajectoryError("episode must be an integer or numeric string")
    if episode is not None:
        rows = [row for row in rows if row.get("episode") in {episode, str(episode)}]
        if not rows:
            raise TrajectoryError(f"no records found for episode={episode}")
    else:
        episode_values = {row.get("episode") for row in rows if row.get("episode") is not None}
        if len(episode_values) > 1:
            raise TrajectoryError(
                f"file contains multiple episodes {sorted(episode_values, key=str)!r}; pass --episode explicitly"
            )

    samples: list[JointSample] = []
    for row_index, row in enumerate(rows):
        # 无时间戳的普通列表仍可用于快速空运行：按默认周期补时间，而不是把所有
        # 记录都压到 t=0 造成互相覆盖。
        if _field(row, "timestamp_s", "timestamp", "time_s") is None:
            row = dict(row)
            row["timestamp_s"] = row_index * default_dt_s
        samples.extend(
            _expand_record(row, use_nested_prediction=use_nested_prediction, default_dt_s=default_dt_s)
        )
    if not samples:
        raise TrajectoryError("trajectory contains no samples")
    samples.sort(key=lambda item: item.timestamp_s)
    # 多个预测窗口可能覆盖同一时刻；保留文件中后展开的记录。Trajectory 构造器
    # 仍会拒绝任何残留的非递增时间，不对有歧义的数据静默猜测。
    unique: list[JointSample] = []
    for sample in samples:
        if unique and math.isclose(sample.timestamp_s, unique[-1].timestamp_s, rel_tol=0.0, abs_tol=1e-9):
            unique[-1] = sample
        else:
            unique.append(sample)
    if normalize_start:
        start = unique[0].timestamp_s
        unique = [
            JointSample(sample.timestamp_s - start, sample.angle_rad, sample.velocity_rad_s, sample.acceleration_rad_s2)
            for sample in unique
        ]
    return Trajectory(tuple(unique))


# 保留初版部署脚手架中的名称，避免已有脚本立即失效。
TrajectoryPoint = JointSample
DEFAULT_JOINT_NAMES = ACTUATED_JOINT_NAMES


__all__ = [
    "ACTUATED_JOINT_NAMES",
    "DEFAULT_JOINT_NAMES",
    "JOINT_COUNT",
    "JointSample",
    "TrajectoryPoint",
    "Trajectory",
    "TrajectoryError",
    "load_trajectory",
]
