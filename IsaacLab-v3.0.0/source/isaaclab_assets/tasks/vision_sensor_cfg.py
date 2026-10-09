"""外骨骼模仿场景的固定左右视觉传感器配置工具。

环境配置在创建场景时调用本模块。输入是左右侧别、通道几何、相机分辨率与更新周期，
输出是 Isaac Lab ``CameraCfg``、EVIS ``DVSCameraCfg`` 或运行时对象。相机位姿、焦距和
事件阈值集中在这里，训练配置与回放配置因此共享同一标定。

Isaac Lab 有渲染 RGB 相机，但没有原生 DVS。事件路径显式依赖可选 EVIS ``dvs_gen``；
未安装时只能创建返回无效空事件的延迟占位配置，真正请求 EVIS 配置/运行时会给出明确
异常，绝不会退化成 RGB 帧差。

所有 prim 路径都使用 ``{ENV_REGEX_NS}``，让每个并行环境在自己的局部通道坐标系中
拥有独立左右相机。若使用世界全局相机，所有环境会共享一幅图，不能作为逐环境 PPO
观测。典型启动顺序是：计算固定 look-at 位姿 -> 构建场景相机配置 -> 创建场景 ->
可选挂接 EVIS 运行时 -> 每个控制周期压缩为固定特征。
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any, Literal

import torch

from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    create_rotation_matrix_from_view,
    quat_from_matrix,
)

# 数据集/仿真契约：用户规格给出了通道长度但没有给出实体宽度，因此半宽保持可配置。
CHANNEL_LENGTH_M = 2.4
CAMERA_X_M = CHANNEL_LENGTH_M / 2.0
CAMERA_Z_M = 1.0
DEFAULT_CHANNEL_HALF_WIDTH_M = 1.2
DEFAULT_TARGET_Z_M = 0.70
CAMERA_RESOLUTION = (256, 256)  # （高度，宽度）
DEFAULT_DVS_THRESHOLD = 0.15
# USD 相机参数以厘米记录。8 cm 镜头配合默认 20.955 cm 水平片门可得到约 104 deg
# 水平 FOV，从侧向偏移 1.2 m 的位置能够覆盖 2.4 m 通道的两端。
CAMERA_FOCAL_LENGTH_CM = 8.0
CAMERA_HORIZONTAL_APERTURE_CM = 20.955

RGB_CAMERA_NAMES = ("left_rgb_camera", "right_rgb_camera")
EVENT_CAMERA_NAMES = ("left_event_camera", "right_event_camera")
ALL_CAMERA_NAMES = RGB_CAMERA_NAMES + EVENT_CAMERA_NAMES


def look_at_quaternion(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    *,
    convention: Literal["opengl", "ros", "world"] = "opengl",
) -> tuple[float, float, float, float]:
    """计算 ``CameraCfg.OffsetCfg`` 使用的 look-at 四元数。

    ``create_rotation_matrix_from_view`` 先生成相机 spawner 所需的 OpenGL world-to-view
    旋转，再按需转换到 ROS 或 world 相机约定。

    Args:
        eye: 相机位置 ``(x,y,z)`` [m]。
        target: 相机注视点 ``(x,y,z)`` [m]。
        convention: 输出坐标约定。

    Returns:
        ``(x,y,z,w)`` 顺序的单位四元数。

    Raises:
        ValueError: eye/target 不有限或两点重合，导致方向无法定义。
    """

    eye_tensor = torch.as_tensor(eye, dtype=torch.float32).reshape(1, 3)
    target_tensor = torch.as_tensor(target, dtype=torch.float32).reshape(1, 3)
    rotation = create_rotation_matrix_from_view(eye_tensor, target_tensor, up_axis="Z", device="cpu")
    if torch.isnan(rotation).any():
        raise ValueError(f"look-at eye and target must be distinct finite points: eye={eye}, target={target}")
    quaternion = quat_from_matrix(rotation)
    if convention != "opengl":
        quaternion = convert_camera_frame_orientation_convention(quaternion, origin="opengl", target=convention)
    return tuple(float(value) for value in quaternion[0].tolist())


def fixed_camera_pose(
    side: Literal["left", "right"],
    *,
    channel_half_width_m: float = DEFAULT_CHANNEL_HALF_WIDTH_M,
    target_z_m: float = DEFAULT_TARGET_Z_M,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """返回通道左侧或右侧固定相机的位置与朝向。

    ``+Y`` 为通道左侧。相机位于通道中点 ``x=1.2 m``、高度 ``1.0 m``，注视通道中线
    上方目标，使躯干和双脚能同时进入视野。

    Args:
        side: ``left`` 或 ``right``。
        channel_half_width_m: 相机离通道中线的横向距离 [m]。
        target_z_m: 注视点高度 [m]。

    Returns:
        ``(position_m, quaternion_xyzw)``。

    Raises:
        ValueError: 侧别非法或半宽不为正数。
    """

    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if channel_half_width_m <= 0.0:
        raise ValueError("channel_half_width_m must be positive")
    y = channel_half_width_m if side == "left" else -channel_half_width_m
    position = (CAMERA_X_M, y, CAMERA_Z_M)
    target = (CAMERA_X_M, 0.0, target_z_m)
    return position, look_at_quaternion(position, target)


def horizontal_fov_deg(
    *, focal_length_cm: float = CAMERA_FOCAL_LENGTH_CM, horizontal_aperture_cm: float = CAMERA_HORIZONTAL_APERTURE_CM
) -> float:
    """由焦距与水平片门尺寸计算针孔相机水平视场角 [deg]。

    两个输入均使用 USD 相机配置的厘米单位；只要单位一致，比例公式不受影响。

    Raises:
        ValueError: 焦距或 aperture 不为正数。
    """

    if focal_length_cm <= 0.0 or horizontal_aperture_cm <= 0.0:
        raise ValueError("focal length and horizontal aperture must be positive")
    return math.degrees(2.0 * math.atan(horizontal_aperture_cm / (2.0 * focal_length_cm)))


def make_rgb_camera_cfg(
    name: str,
    side: Literal["left", "right"],
    *,
    channel_half_width_m: float = DEFAULT_CHANNEL_HALF_WIDTH_M,
    target_z_m: float = DEFAULT_TARGET_Z_M,
    update_period: float = 0.02,
    width: int = CAMERA_RESOLUTION[1],
    height: int = CAMERA_RESOLUTION[0],
) -> Any:
    """构建每环境独立的固定 RGB ``CameraCfg``。

    ``name`` 与 ``side`` 分离，因此同一函数可创建环境要求的左右命名实体。相机固定在
    每个环境原点下，不随机器人 articulation 移动。

    Args:
        name: 场景实体/prim 名称。
        side: 通道左侧或右侧。
        channel_half_width_m: 相机横向距离 [m]。
        target_z_m: 注视点高度 [m]。
        update_period: 图像更新周期 [s]。
        width: 输出宽度 [pixel]。
        height: 输出高度 [pixel]。

    Returns:
        data type 仅包含 ``rgb`` 的 Isaac Lab ``CameraCfg``。
    """

    from isaaclab import sim as sim_utils
    from isaaclab.sensors import CameraCfg

    position, rotation = fixed_camera_pose(side, channel_half_width_m=channel_half_width_m, target_z_m=target_z_m)
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(pos=position, rot=rotation, convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH_CM,
            focus_distance=3.0,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE_CM,
            clipping_range=(0.05, 10.0),
        ),
    )


def event_camera_plugin_available() -> bool:
    """仅检查可选 EVIS ``dvs_gen`` 是否可导入，不创建相机或运行时。"""

    return importlib.util.find_spec("dvs_gen") is not None


def make_event_camera_cfg(
    name: str,
    side: Literal["left", "right"],
    *,
    channel_half_width_m: float = DEFAULT_CHANNEL_HALF_WIDTH_M,
    target_z_m: float = DEFAULT_TARGET_Z_M,
    threshold: float = DEFAULT_DVS_THRESHOLD,
    update_period: float = 0.02,
    width: int = CAMERA_RESOLUTION[1],
    height: int = CAMERA_RESOLUTION[0],
) -> Any:
    """用与 RGB 相机相同位姿构建 EVIS ``DVSCameraCfg``。

    模块导入时不强制 EVIS，从而让纯 Isaac Lab 环境仍可查看帮助和配置。真正训练事件
    观测必须安装 EVIS 并使用其 ``DVSCamera`` 运行时；普通 Camera 无法产生异步
    ``(x,y,t,polarity)`` 事件。

    Args:
        name: 场景实体名称。
        side: ``left`` 或 ``right``。
        threshold: DVS 对比度触发阈值，无量纲。
        update_period: 渲染/事件更新周期 [s]。
        width: 渲染宽度 [pixel]。
        height: 渲染高度 [pixel]。

    Returns:
        请求 RGB/HDR、运动向量和深度缓冲的 EVIS 配置。

    Raises:
        RuntimeError: 未安装 EVIS ``dvs_gen``。
        ValueError: 固定相机几何参数不合法。
    """

    try:
        from dvs_gen.sensors import DVSCameraCfg
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Event camera support requires the EVIS `dvs_gen` plugin. "
            "Install it into the Isaac Sim Python environment from "
            "https://github.com/spikelab-jhu/isaac-sim-event-camera-plugin; "
            "RGB frame differencing is intentionally not used as a fallback."
        ) from exc

    from isaaclab import sim as sim_utils

    position, rotation = fixed_camera_pose(side, channel_half_width_m=channel_half_width_m, target_z_m=target_z_m)
    return DVSCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        update_period=update_period,
        height=height,
        width=width,
        threshold=threshold,
        event_source="hdr",
        enable_warp=True,
        data_types=["rgb", "HdrColor", "motion_vectors", "distance_to_image_plane"],
        offset=DVSCameraCfg.OffsetCfg(pos=position, rot=rotation, convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH_CM,
            focus_distance=3.0,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE_CM,
            clipping_range=(0.05, 10.0),
        ),
    )


def make_deferred_event_camera_cfg(
    name: str,
    side: Literal["left", "right"],
    *,
    channel_half_width_m: float = DEFAULT_CHANNEL_HALF_WIDTH_M,
    target_z_m: float = DEFAULT_TARGET_Z_M,
    threshold: float = DEFAULT_DVS_THRESHOLD,
    update_period: float = 0.02,
    width: int = CAMERA_RESOLUTION[1],
    height: int = CAMERA_RESOLUTION[0],
) -> Any:
    """构建未安装 EVIS 时仅供配置/冒烟测试使用的占位相机。

    占位器创建相同固定相机 prim，通过 ``EventCamera`` 暴露空且无效的事件流。它让
    IMU/RGB 测试可运行，但不是事件仿真，也不会从 RGB 差分生成伪事件。

    Args:
        name: 场景实体名称。
        side: 通道左侧或右侧。
        channel_half_width_m: 相机横向距离 [m]。
        target_z_m: 注视点高度 [m]。
        threshold: 保存到配置、供未来 EVIS 使用的对比度阈值。
        update_period: 渲染更新周期 [s]。
        width: 渲染宽度 [pixel]。
        height: 渲染高度 [pixel]。

    Returns:
        Isaac Lab 可创建的 ``EventCameraCfg``；其中事件有效性始终由运行时决定。
    """

    from isaaclab import sim as sim_utils

    from isaaclab_assets.sensors.event_camera import EventCameraCfg

    position, rotation = fixed_camera_pose(side, channel_half_width_m=channel_half_width_m, target_z_m=target_z_m)
    return EventCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        update_period=update_period,
        height=height,
        width=width,
        # 保留渲染器缓冲供已挂接的 EVIS 使用；此适配器自身绝不会把缓冲伪造成事件。
        data_types=["rgb", "motion_vectors", "distance_to_image_plane"],
        contrast_threshold=threshold,
        offset=EventCameraCfg.OffsetCfg(pos=position, rot=rotation, convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH_CM,
            focus_distance=3.0,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE_CM,
            clipping_range=(0.05, 10.0),
        ),
    )


def make_fixed_camera_cfgs(
    *,
    include_event: bool = False,
    channel_half_width_m: float = DEFAULT_CHANNEL_HALF_WIDTH_M,
    target_z_m: float = DEFAULT_TARGET_Z_M,
    threshold: float = DEFAULT_DVS_THRESHOLD,
) -> dict[str, Any]:
    """批量构建左右 RGB 配置，并按需加入左右 EVIS 配置。

    Args:
        include_event: 是否同时构建两路真实 EVIS 配置。
        channel_half_width_m: 相机离通道中线距离 [m]。
        target_z_m: 注视点高度 [m]。
        threshold: DVS 对比度阈值。

    Returns:
        以 ``left/right_rgb_camera`` 命名的字典；``include_event=True`` 时再包含两路
        ``*_event_camera``。请求事件但未安装 EVIS 时会传播 :class:`RuntimeError`。
    """

    configs: dict[str, Any] = {
        "left_rgb_camera": make_rgb_camera_cfg(
            "left_rgb_camera", "left", channel_half_width_m=channel_half_width_m, target_z_m=target_z_m
        ),
        "right_rgb_camera": make_rgb_camera_cfg(
            "right_rgb_camera", "right", channel_half_width_m=channel_half_width_m, target_z_m=target_z_m
        ),
    }
    if include_event:
        configs.update(
            {
                "left_event_camera": make_event_camera_cfg(
                    "left_event_camera",
                    "left",
                    channel_half_width_m=channel_half_width_m,
                    target_z_m=target_z_m,
                    threshold=threshold,
                ),
                "right_event_camera": make_event_camera_cfg(
                    "right_event_camera",
                    "right",
                    channel_half_width_m=channel_half_width_m,
                    target_z_m=target_z_m,
                    threshold=threshold,
                ),
            }
        )
    return configs


def make_dvs_runtime(scene: Any, names: tuple[str, ...] = EVENT_CAMERA_NAMES, *, out_dir: str) -> Any:
    """为已经建好的场景相机创建 EVIS 异步事件运行时。

    返回对象提供 ``snapshot/process/warp_and_process`` 并记录真实异步事件。变长事件流
    不能直接放进 manager-based 批量观测，环境会先分箱/池化为固定 32 维特征。这里还
    为每个 processor 开启 ``stash_events``，让环境既可读取小掩码，记录器又能保留
    原始事件。

    Args:
        scene: 已完成相机实体创建的 Isaac Lab 场景。
        names: 要接管的事件相机实体名，默认左右两路。
        out_dir: EVIS 原始事件输出目录。

    Returns:
        EVIS ``DVSCamera`` 运行时对象。

    Raises:
        RuntimeError: 未安装 EVIS；其他 EVIS 场景匹配错误由插件原样抛出。
    """

    try:
        from dvs_gen import DVSCamera
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Event runtime requires the EVIS `dvs_gen` plugin; no RGB-difference fallback is provided."
        ) from exc
    runtime = DVSCamera.from_scene(scene, names, out_dir=out_dir, event_source="hdr")
    # EVIS 把异步掩码保存在各相机 processor 上。启用这块小缓冲后，环境可以向 PPO 提供
    # 固定直方图，同时记录器仍能收到原始事件流。
    for processor in getattr(runtime, "procs", ()):
        processor.stash_events = True
    return runtime


__all__ = [
    "ALL_CAMERA_NAMES",
    "CAMERA_RESOLUTION",
    "CAMERA_FOCAL_LENGTH_CM",
    "CAMERA_HORIZONTAL_APERTURE_CM",
    "CHANNEL_LENGTH_M",
    "DEFAULT_CHANNEL_HALF_WIDTH_M",
    "DEFAULT_DVS_THRESHOLD",
    "EVENT_CAMERA_NAMES",
    "RGB_CAMERA_NAMES",
    "event_camera_plugin_available",
    "fixed_camera_pose",
    "horizontal_fov_deg",
    "look_at_quaternion",
    "make_dvs_runtime",
    "make_deferred_event_camera_cfg",
    "make_event_camera_cfg",
    "make_fixed_camera_cfgs",
    "make_rgb_camera_cfg",
]
