"""外骨骼模仿任务的可选事件相机（Event Camera）适配层。

Isaac Lab 3.0 本身没有 DVS（Dynamic Vision Sensor，动态视觉传感器）。外骨骼
环境配置先把 :class:`EventCameraCfg` 当作普通渲染相机创建；环境启动后，
``vision_sensor_cfg.make_dvs_runtime`` 可再把 EVIS ``dvs_gen`` 运行时接入本类。

输入包括相机标定配置和可选 EVIS 运行时，输出 :class:`EventBatch`。适配器约定
运行时返回的事件张量最后一维是 ``(x, y, t, polarity)``，但本类只透传运行时结果，
不会自动验证该顺序、shape 或数值范围。当前 PPO 观测主路径并不调用 `event_batch()`，
而是直接读取 EVIS processor 的 `last_masks` 正负极性平面并池化成固定特征。
若 EVIS 未安装，本模块仍允许 IMU/RGB 等其他模态训练，但明确返回“空且无效”的
事件流，不会用相邻 RGB 帧差伪造事件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.configclass import configclass


@configclass
class EventCameraCfg(CameraCfg):
    """在普通相机配置上增加 DVS 生成参数。

    Attributes:
        class_type: Isaac Lab 创建传感器时动态导入的实现类。
        contrast_threshold: 触发亮度事件的对比度阈值，无量纲。
        event_source: 交给 EVIS 的图像源名称。
        enable_warp: 是否允许运行时使用 Warp 加速。
        event_width: 输出事件网格宽度 [pixel]。
        event_height: 输出事件网格高度 [pixel]。
    """

    class_type: Any = "isaaclab_assets.sensors.event_camera:EventCamera"
    contrast_threshold: float = 0.15
    event_source: str = "hdr"
    enable_warp: bool = True
    event_width: int = 64
    event_height: int = 64


@dataclass
class EventBatch:
    """按运行时适配器约定，包装变长事件流与有效掩码。

    Attributes:
        events: 事件数据，适配器应提供 shape
            ``[num_envs, num_events, 4]``，并把最后一维排成 ``(x, y, t, polarity)``。
            本数据类不执行这些校验；没有运行时时 ``num_events`` 为 ``0``。
        valid: shape 为 ``[num_envs]`` 的布尔张量；``False`` 表示该环境没有可信事件。
    """

    events: torch.Tensor
    valid: torch.Tensor


class EventCamera(Camera):
    """提供真实事件接口、同时保留渲染后端的相机。

    实际异步事件模型来自运行时挂接的 EVIS。在纯 Isaac Lab 安装中，
    :meth:`event_batch` 返回空且无效的流，让不使用事件相机的冒烟测试可以运行，
    同时不会悄悄改变“事件”这一数据语义。
    """

    cfg: EventCameraCfg

    def __init__(self, cfg: EventCameraCfg):
        """创建渲染相机，并预留尚未挂接的事件运行时。

        Args:
            cfg: 相机路径、成像参数和 DVS 参数配置。
        """

        super().__init__(cfg)
        self._event_runtime: Any | None = None

    def attach_runtime(self, runtime: Any) -> None:
        """挂接场景创建后生成的 EVIS ``DVSCamera`` 运行时。

        Args:
            runtime: 应实现 ``events_for(prim_path)`` 的对象；该方法应返回
                ``(events, valid)`` 两个张量。

        Returns:
            无返回值；后续 :meth:`event_batch` 会从该对象读取事件。
        """

        self._event_runtime = runtime

    def event_batch(self) -> EventBatch:
        """按适配器约定读取当前环境批次的事件。

        EVIS 原始事件流长度可变，运行时桥接层可以返回已分箱的固定 shape 张量。
        已挂接运行时时，本方法原样包装 ``events_for`` 的两个返回值，不校验事件列顺序、
        shape、dtype 或有效掩码长度；这些都是运行时适配器的前置契约。
        没有运行时时，环境数量优先从传感器 view 获取，再回退到 prim 数量；随后在
        相机设备上创建 shape ``[num_envs, 0, 4]`` 的空事件和 ``[num_envs]`` 无效掩码。

        Returns:
            :class:`EventBatch`。按适配器契约，事件坐标和时间戳尺度由 EVIS 标定负责，
            ``polarity`` 表示亮度变化方向。
        """

        view = getattr(self, "_view", None)
        count = int(getattr(view, "count", 0) or getattr(self, "_num_envs", 0))
        if count == 0:
            prim_paths = getattr(self, "prim_paths", ())
            count = len(prim_paths)
        device = getattr(self, "device", torch.device("cpu"))
        if self._event_runtime is not None and hasattr(self._event_runtime, "events_for"):
            events, valid = self._event_runtime.events_for(self.cfg.prim_path)
            return EventBatch(events=events, valid=valid)
        return EventBatch(
            events=torch.zeros((count, 0, 4), dtype=torch.float32, device=device),
            valid=torch.zeros((count,), dtype=torch.bool, device=device),
        )


__all__ = ["EventBatch", "EventCamera", "EventCameraCfg"]
