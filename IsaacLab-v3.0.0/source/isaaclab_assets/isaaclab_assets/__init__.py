# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Isaac Lab 资产扩展包入口。

本文件主要公布扩展根目录、数据目录和版本号。外骨骼任务源码按本项目布局放在扩展根
目录的 tasks/envs 中，因此源码运行时会把该目录追加到包搜索路径；这不是训练逻辑。
"""

import importlib.metadata
import os
from pathlib import Path

ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""扩展源码根目录的绝对路径。"""

ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""机器人 URDF、网格等资产数据目录的绝对路径。"""

# 让位于扩展根目录的自定义任务/配置模块在源码检出目录中仍可导入，这是项目布局指定的
# 位置。安装后的 wheel 应像往常一样把这些模块打包到 ``isaaclab_assets`` 下。
# 新手提示：这里扩展的是 Python 的包搜索路径，不是 Linux 的 PATH 环境变量。
# 删除它会导致源码工作区中 import isaaclab_assets.tasks 失败。
_extension_root = str(Path(ISAACLAB_ASSETS_EXT_DIR))
if _extension_root not in __path__:
    __path__.append(_extension_root)


try:
    __version__ = importlib.metadata.version("isaaclab_assets")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from isaaclab.utils.module import lazy_export

lazy_export()
