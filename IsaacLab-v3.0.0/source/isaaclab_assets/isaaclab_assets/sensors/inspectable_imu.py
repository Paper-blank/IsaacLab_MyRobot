# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""为 PhysX IMU 创建可在 Kit 中点击查看的实时探针。

外骨骼场景通过 ``ImuCfg.class_type`` 创建 :class:`InspectableImu`。输入仍是普通 IMU
配置和每个仿真步产生的 IMU 数据；输出仍由父类提供，同时在开启 ``debug_vis`` 时，
本类会为每个 IMU 源 prim 建立一个非实例化 USD 球体，并把实时读数写入 ``imu:*``
自定义属性，供用户在 Kit 的 Properties 面板查看。

启动顺序是：PhysX IMU 初始化 -> 用户/回放脚本开启调试显示 -> 创建一次探针 prim ->
可视化回调以最高 10 Hz 刷新属性。探针只用于人工诊断，不进入策略观测，也不会改变
传感器原始张量。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch
from isaaclab_physx.sensors.imu import Imu as PhysxImu

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.sensors import ImuCfg

logger = logging.getLogger(__name__)


class InspectableImu(PhysxImu):
    """带可点击 USD 探针的 PhysX IMU 调试实现。

    Isaac Lab 常规可视化标记使用 point instancing，Kit 只能选中整组标记。这里为每个
    IMU 目标创建普通 USD 球体，所以可在视口或 Stage 树中逐个选择并查看实时属性。
    """

    _MARKER_NAME = "IMU_Inspector"
    _TELEMETRY_UPDATE_PERIOD_S = 0.1

    def __init__(self, cfg: ImuCfg):
        """初始化探针状态，再让父类创建实际 IMU。

        Args:
            cfg: Isaac Lab IMU 配置；``prim_path`` 可匹配多个克隆环境中的肢段。
        """

        self._telemetry_prims: list[Any] = []
        self._last_telemetry_update_s = float("-inf")
        super().__init__(cfg)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """按调试开关创建、显示或隐藏可点击探针。

        Args:
            debug_vis: ``True`` 时确保探针存在并可见，``False`` 时只隐藏已有探针。

        Returns:
            无返回值；副作用发生在当前内存 USD Stage。
        """
        if debug_vis:
            self._ensure_telemetry_prims()
        if not self._telemetry_prims:
            return

        from pxr import UsdGeom  # noqa: PLC0415

        for prim in self._telemetry_prims:
            imageable = UsdGeom.Imageable(prim)
            if debug_vis:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

    def _debug_vis_callback(self, event) -> None:
        """把最新 IMU 读数发布到探针的 USD 属性。

        Args:
            event: Kit 可视化回调事件。本实现不读取其内容，只用它作为回调签名。

        Returns:
            无返回值。最多每 ``0.1 s`` 更新一次，避免每个渲染帧都进行 GPU 到 CPU
            拷贝和 USD 属性写入。

        Notes:
            ``ang_vel_b`` 和 ``lin_acc_b`` 的 shape 均为 ``[num_sensors, 3]``，分别使用
            ``rad/s`` 与 ``m/s^2``，坐标系为各传感器的 body frame。
        """
        del event
        if getattr(self, "_view", None) is None or not self._telemetry_prims:
            return

        now = time.monotonic()
        if now - self._last_telemetry_update_s < self._TELEMETRY_UPDATE_PERIOD_S:
            return

        data = self.data
        angular_velocity = self._to_torch(data.ang_vel_b)
        linear_acceleration = self._to_torch(data.lin_acc_b)
        count = min(len(self._telemetry_prims), angular_velocity.shape[0], linear_acceleration.shape[0])
        if count != len(self._telemetry_prims):
            logger.warning(
                "IMU probe count (%d) does not match sensor data (%d); publishing the shared prefix.",
                len(self._telemetry_prims),
                min(angular_velocity.shape[0], linear_acceleration.shape[0]),
            )

        from pxr import Gf  # noqa: PLC0415

        # USD 的 Gf 属性在 CPU 侧写入，因此先切断 autograd 并只复制实际可发布的前缀。
        angular_velocity = angular_velocity[:count].detach().cpu()
        linear_acceleration = linear_acceleration[:count].detach().cpu()
        for index, prim in enumerate(self._telemetry_prims[:count]):
            prim.GetAttribute("imu:angularVelocity").Set(Gf.Vec3f(*angular_velocity[index].tolist()))
            prim.GetAttribute("imu:linearAcceleration").Set(Gf.Vec3f(*linear_acceleration[index].tolist()))
        self._last_telemetry_update_s = now

    def _ensure_telemetry_prims(self) -> None:
        """在每个 IMU 源 prim 下创建一次非实例化可见球体。

        已经创建过时直接返回。若路径没有匹配目标，只记录警告，以免纯数据传感器因为
        可视化失败而中断训练或回放。

        Returns:
            无返回值；成功时 ``self._telemetry_prims`` 与匹配到的源 prim 一一对应。
        """
        if self._telemetry_prims:
            return

        from pxr import Gf, Sdf, UsdGeom  # noqa: PLC0415

        source_prims = sim_utils.find_matching_prims(self.cfg.prim_path, stage=self.stage)
        if not source_prims:
            logger.warning("No source prims found for IMU debug visualization at '%s'.", self.cfg.prim_path)
            return

        sensor_name = self.cfg.prim_path.rsplit("/", 1)[-1]
        for source_prim in source_prims:
            marker_path = f"{source_prim.GetPath()}/{self._MARKER_NAME}"
            marker = UsdGeom.Sphere.Define(self.stage, marker_path)
            marker_prim = marker.GetPrim()
            marker_prim.SetDisplayName(f"IMU {sensor_name} (click for telemetry)")
            marker.CreateRadiusAttr(0.06)
            marker.CreateDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.7, 1.0)])
            marker.AddTranslateOp().Set(Gf.Vec3d(*self.cfg.offset.pos))

            self._create_attribute(marker_prim, "imu:angularVelocity", Sdf.ValueTypeNames.Vector3f)
            self._create_attribute(marker_prim, "imu:linearAcceleration", Sdf.ValueTypeNames.Vector3f)
            self._create_attribute(marker_prim, "imu:sourcePrim", Sdf.ValueTypeNames.String).Set(
                source_prim.GetPath().pathString
            )
            self._create_attribute(marker_prim, "imu:frame", Sdf.ValueTypeNames.String).Set("body")
            self._create_attribute(marker_prim, "imu:angularVelocityUnit", Sdf.ValueTypeNames.String).Set("rad/s")
            self._create_attribute(marker_prim, "imu:linearAccelerationUnit", Sdf.ValueTypeNames.String).Set("m/s^2")
            self._telemetry_prims.append(marker_prim)

    @staticmethod
    def _create_attribute(prim: Any, name: str, value_type: Any):
        """获取一个 USD 自定义属性，不存在时就创建。

        Args:
            prim: 承载属性的 USD prim。
            name: 完整属性名，例如 ``imu:angularVelocity``。
            value_type: ``pxr.Sdf.ValueTypeNames`` 中的 USD 数据类型。

        Returns:
            有效的 ``Usd.Attribute``；重复调用同一属性不会重复创建。
        """
        attribute = prim.GetAttribute(name)
        if not attribute.IsValid():
            attribute = prim.CreateAttribute(name, value_type, custom=True)
        return attribute

    @staticmethod
    def _to_torch(value: Any) -> torch.Tensor:
        """把 Isaac Lab 代理数组统一成 Torch 张量。

        Args:
            value: 原生 ``torch.Tensor``，或带 ``.torch`` 属性的后端代理数组。

        Returns:
            与输入共享语义、shape 和设备的张量；本函数不复制或改单位。
        """
        return value.torch if hasattr(value, "torch") else value
