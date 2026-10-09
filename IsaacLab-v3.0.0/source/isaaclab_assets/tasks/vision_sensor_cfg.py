"""Fixed side-camera helpers for the exoskeleton imitation scene.

Isaac Lab ships a renderer-backed :class:`CameraCfg`, but it does not ship an
event-camera sensor.  This module keeps the placement/calibration contract in
one place and makes the optional EVIS (``dvs_gen``) dependency explicit.  The
event-camera helper never falls back to frame differencing: without EVIS the
caller gets an actionable error instead of silently training on a different
sensor model.

The camera paths intentionally use ``{ENV_REGEX_NS}``.  This gives every
parallel environment its own fixed side rig in its local channel frame.  A
world-global camera would produce one image shared by all environments and is
therefore unsuitable as a per-environment PPO observation.
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

# Dataset/simulation contract.  The channel half-width is configurable because
# the user specification gives its length but not its physical width.
CHANNEL_LENGTH_M = 2.4
CAMERA_X_M = CHANNEL_LENGTH_M / 2.0
CAMERA_Z_M = 1.0
DEFAULT_CHANNEL_HALF_WIDTH_M = 1.2
DEFAULT_TARGET_Z_M = 0.70
CAMERA_RESOLUTION = (256, 256)  # (height, width)
DEFAULT_DVS_THRESHOLD = 0.15
# USD camera values are authored in centimetres.  An 8 cm lens gives about
# 104 degrees horizontal FOV with the default 20.955 cm film gate, which
# covers both ends of the 2.4 m channel from the 1.2 m side offset.
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
    """Return an ``(x, y, z, w)`` look-at quaternion for ``CameraCfg.OffsetCfg``.

    ``create_rotation_matrix_from_view`` returns the world-to-view OpenGL
    rotation expected by Isaac Lab's camera spawner.  The optional conversion
    is applied only after the look-at calculation, so callers can request the
    same pose in ROS or world convention when needed.
    """

    eye_tensor = torch.as_tensor(eye, dtype=torch.float32).reshape(1, 3)
    target_tensor = torch.as_tensor(target, dtype=torch.float32).reshape(1, 3)
    rotation = create_rotation_matrix_from_view(eye_tensor, target_tensor, up_axis="Z", device="cpu")
    if torch.isnan(rotation).any():
        raise ValueError(f"look-at eye and target must be distinct finite points: eye={eye}, target={target}")
    quaternion = quat_from_matrix(rotation)
    if convention != "opengl":
        quaternion = convert_camera_frame_orientation_convention(
            quaternion, origin="opengl", target=convention
        )
    return tuple(float(value) for value in quaternion[0].tolist())


def fixed_camera_pose(
    side: Literal["left", "right"],
    *,
    channel_half_width_m: float = DEFAULT_CHANNEL_HALF_WIDTH_M,
    target_z_m: float = DEFAULT_TARGET_Z_M,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return the fixed side pose and look-at rotation.

    The left/right sign follows the conventional +Y left-hand side of the
    channel.  The target is the channel centre at ``x=1.2``; keeping it above
    the floor gives the robot's torso and feet useful pixels at once.
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
    """Return the pinhole horizontal field of view [deg]."""

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
    """Build a fixed RGB :class:`isaaclab.sensors.CameraCfg`.

    ``name`` is kept separate from ``side`` so the same helper can construct
    both the left and right named sensors required by the environment config.
    The camera is fixed relative to each environment origin; it is not attached
    to the robot articulation.
    """

    from isaaclab import sim as sim_utils
    from isaaclab.sensors import CameraCfg

    position, rotation = fixed_camera_pose(
        side, channel_half_width_m=channel_half_width_m, target_z_m=target_z_m
    )
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
    """Return whether the optional EVIS ``dvs_gen`` package can be imported."""

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
    """Build an EVIS ``DVSCameraCfg`` with the same fixed pose as RGB.

    The dependency is intentionally optional at import time so config/help
    commands remain usable on a stock Isaac Lab installation.  Training with
    an event observation must install EVIS and use its runtime ``DVSCamera``
    wrapper; an Isaac Lab ``Camera`` alone cannot emit asynchronous
    ``(x, y, t, polarity)`` events.
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

    position, rotation = fixed_camera_pose(
        side, channel_half_width_m=channel_half_width_m, target_z_m=target_z_m
    )
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
    """Build the Isaac-Lab-only placeholder used before EVIS is installed.

    The placeholder creates the same fixed camera prim and exposes an empty
    invalid event stream through :class:`EventCamera`.  It is useful for
    configuration and IMU/RGB smoke tests; it is not an event simulation and
    never derives events from RGB differences.
    """

    from isaaclab import sim as sim_utils

    from isaaclab_assets.sensors.event_camera import EventCameraCfg

    position, rotation = fixed_camera_pose(
        side, channel_half_width_m=channel_half_width_m, target_z_m=target_z_m
    )
    return EventCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        update_period=update_period,
        height=height,
        width=width,
        # The renderer is retained for EVIS to consume when it is attached;
        # this adapter itself never turns these buffers into fake events.
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
    """Return the required left/right RGB pair and optional DVS pair."""

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
    """Create EVIS' runtime event pipeline for already-built scene cameras.

    The returned object exposes ``snapshot()``, ``process()`` or
    ``warp_and_process()`` and records true asynchronous events.  It is kept
    out of manager-based observations because event streams are ragged; callers
    should bin them into a fixed histogram/voxel tensor before adding them to
    PPO observations.
    """

    try:
        from dvs_gen import DVSCamera
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Event runtime requires the EVIS `dvs_gen` plugin; no RGB-difference "
            "fallback is provided."
        ) from exc
    runtime = DVSCamera.from_scene(scene, names, out_dir=out_dir, event_source="hdr")
    # EVIS keeps the asynchronous masks on each per-camera processor.  Opt in
    # to that small buffer so the environment can expose a fixed histogram for
    # PPO while the recorder still receives the original event stream.
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
