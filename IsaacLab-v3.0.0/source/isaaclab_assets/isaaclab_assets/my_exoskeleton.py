"""把两个外骨骼环境注册到 Gymnasium。

训练脚本和回放脚本都会在调用 Isaac Lab 通用入口前导入本模块。导入本身就是启动动作：
模块把普通平地运动任务和多模态模仿任务的 ID、环境类、环境配置和 RL-Games YAML
绑定起来。之后 ``gym.make`` 或 Isaac Lab 的配置加载器才能通过字符串 ID 找到它们。

本模块没有运行时输入文件，也不创建环境；输入是下方固定的注册元数据，输出是
``gym.registry`` 中的两条记录。注册前会检查 ID 是否已存在，因此同一进程重复导入
不会重复注册。实际启动顺序是：导入本模块 -> 解析任务配置 -> 创建仿真/环境 ->
训练或回放。
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

TASK_ID = "Isaac-MyExoskeleton-Flat-v0"
IMITATION_TASK_ID = "Isaac-MyExoskeleton-Imitation-v0"
# YAML 路径必须使用绝对路径，避免训练后端切换工作目录后找不到 PPO 配置。
_source_root = Path(__file__).resolve().parents[1]
_cfg_entry_point = "isaaclab_assets.envs.my_exoskeleton_env_cfg:MyExoskeletonFlatEnvCfg"
_ppo_entry_point = str(_source_root / "configs" / "my_exoskeleton_ppo.yaml")
_imitation_cfg_entry_point = (
    "isaaclab_assets.envs.my_exoskeleton_imitation_env_cfg:MyExoskeletonImitationEnvCfg"
)
_imitation_env_entry_point = "isaaclab_assets.tasks.my_exoskeleton_imitation:MyExoskeletonImitationEnv"
_imitation_ppo_entry_point = str(_source_root / "configs" / "my_exoskeleton_imitation_ppo.yaml")

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab_assets.tasks.my_exoskeleton_flat:MyExoskeletonFlatEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _cfg_entry_point,
            "rl_games_cfg_entry_point": _ppo_entry_point,
            "default_agent": "rl_games",
        },
    )

if IMITATION_TASK_ID not in gym.registry:
    gym.register(
        id=IMITATION_TASK_ID,
        entry_point=_imitation_env_entry_point,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _imitation_cfg_entry_point,
            "rl_games_cfg_entry_point": _imitation_ppo_entry_point,
            "default_agent": "rl_games",
        },
    )


__all__ = ["IMITATION_TASK_ID", "TASK_ID"]
