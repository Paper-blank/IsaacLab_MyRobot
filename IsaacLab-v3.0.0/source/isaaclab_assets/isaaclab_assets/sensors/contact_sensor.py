# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""为 URDF 转换后的嵌套刚体补齐接触传感器初始化。

外骨骼环境配置通过 ``class_type`` 字符串创建 :class:`NestedBodyContactSensor`。
输入是普通 Isaac Lab 接触传感器配置，其中 ``prim_path`` 可包含环境克隆通配表达式；
输出仍是父类提供的标准接触力/接触历史数据，本模块不会改变张量 shape 或物理单位。

初始化顺序是：Isaac Sim 完成 USD 引用组合 -> 本类找到传感器目标的父级子树并挂载
``PhysxContactReportAPI`` -> 调用 PhysX 原生接触传感器初始化。之所以多做这一步，是
因为 URDF 3.0 转换器把碰撞刚体放进嵌套引用，早期的统一激活回调可能看不到它们。
"""

from __future__ import annotations

import logging

from isaaclab_physx.sensors.contact_sensor import ContactSensor as PhysxContactSensor

from isaaclab import sim as sim_utils
from isaaclab.sim.schemas import schemas

_LOGGER = logging.getLogger(__name__)


class NestedBodyContactSensor(PhysxContactSensor):
    """自动修复缺失接触报告 schema 的 PhysX 接触传感器。

    Isaac Sim 的 URDF 3.0 转换器会把视觉和碰撞实例放在 link prim 下。Spawner 层的
    ``activate_contact_sensors`` 可能在引用刚体完成组合前运行，导致传感器建立视图时
    找不到 ``PhysxContactReportAPI``。本类只在父类初始化前补 schema，因而也适用于
    多环境克隆，并保持父类公开接口不变。
    """

    def _initialize_impl(self):
        """先激活嵌套刚体的接触报告，再执行父类初始化。

        Returns:
            父类 ``_initialize_impl`` 的返回值；具体类型由当前 Isaac Lab/PhysX 版本决定。
        """

        self._activate_nested_contact_reports()
        return super()._initialize_impl()

    def _activate_nested_contact_reports(self) -> None:
        """为路径表达式匹配到的每个父级子树启用接触报告。

        ``self.cfg.prim_path`` 的最后一段是目标刚体表达式，前面的部分用于找到每个
        克隆环境的父 prim。路径解析或 USD schema 操作失败时这里只记录警告，随后仍让
        父类初始化，以便父类给出更具体的刚体/视图错误。

        Returns:
            无返回值；副作用是修改当前内存 Stage 上匹配 prim 的 PhysX schema。
        """

        try:
            segments = sim_utils.split_path_expr(self.cfg.prim_path)
            parent_expr = "/".join(segments[:-1])
            parents = sim_utils.find_matching_prims(parent_expr, stage=self.stage)
            if not parents:
                _LOGGER.warning("No contact-sensor parent matched %r", parent_expr)
                return
            for parent in parents:
                schemas.activate_contact_sensors(parent.GetPath().pathString, stage=self.stage)
        except (RuntimeError, TypeError, ValueError) as exc:
            # 不吞掉父类之后的初始化错误；此处警告只保留“补 schema”阶段的上下文。
            _LOGGER.warning("Unable to activate nested contact reporting for %r: %s", self.cfg.prim_path, exc)


__all__ = ["NestedBodyContactSensor"]
