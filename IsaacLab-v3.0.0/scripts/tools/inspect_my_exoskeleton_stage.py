"""只读检查外骨骼模仿场景最终生成的 USD 路径。

这个脚本供排查“配置里的传感器路径为什么匹配不到实体”等场景装配问题使用，调用者
通常是开发者从终端手动运行。它不会训练策略、推进物理仿真或写回 USD，只会创建一个
``ManagerBasedRLEnv``，读取内存中的 Stage，然后把刚体、关节、克隆计划和传感器信息
打印到标准输出。例如：

    ./isaaclab.sh -p scripts/tools/inspect_my_exoskeleton_stage.py --num_envs 1 --headless

输入是 ``--task``、``--num_envs`` 和 ``AppLauncher`` 的通用参数，输出是便于人工核对的
文本报告。启动顺序不能颠倒：先解析参数并创建 Isaac Sim 应用，再导入 Gym、USD 和
Isaac Lab 环境模块；最后无论检查成功与否都关闭 ``simulation_app``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# AppLauncher 必须先启动 Kit，下面标有 E402 的仿真模块才能安全导入。
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", default="Isaac-MyExoskeleton-Imitation-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[inspect] app launched", flush=True)

repo_root = Path(__file__).resolve().parents[2]
for source_dir in (repo_root / "source" / "isaaclab_assets", repo_root / "source" / "isaaclab_tasks"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import gymnasium as gym  # noqa: E402,I001
import isaaclab_assets.my_exoskeleton  # noqa: E402,F401
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.dict import class_to_dict  # noqa: E402,F401


def main() -> None:
    """构建一次环境，并打印第一个克隆环境的 USD/传感器结构。

    函数读取模块顶层已经解析好的 ``args_cli``。它会将 Gym 注册项中的环境配置实例化，
    把环境数和设备替换为命令行值，然后检查尚未开始物理步进的 Stage。

    Returns:
        无返回值；检查结果逐行写到标准输出。

    Raises:
        gymnasium.error.Error: ``--task`` 未注册或没有对应环境配置。
        Exception: Isaac Sim 场景生成、USD 遍历或传感器初始化失败时原样抛出。
    """

    print("[inspect] entering main", flush=True)
    env_spec = gym.spec(args_cli.task)
    cfg = env_spec.kwargs["env_cfg_entry_point"]
    if isinstance(cfg, str):
        from isaaclab.utils import load_cfg_from_registry

        cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=cfg)
    print("[inspect] env constructed", flush=True)
    stage = sim_utils.get_current_stage()

    from pxr import UsdPhysics

    print("[inspect] task:", args_cli.task)
    print("[inspect] env prims:")
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path.startswith("/World/envs/env_0/Robot") and (
            prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ):
            print("  ", path, prim.GetAppliedSchemas())

    clone_plan = env.sim.get_clone_plan()
    if clone_plan is not None:
        print("[inspect] clone plan:")
        for source, destination, mask in zip(clone_plan.sources, clone_plan.destinations, clone_plan.clone_mask):
            print("  ", source, "->", destination, "envs", mask.nonzero().flatten().tolist())

    print("[inspect] articulations:")
    for name, robot in env.scene.articulations.items():
        print("  ", name, "prim_path=", robot.cfg.prim_path)
        print("      body_names=", list(robot.body_names))
        print(
            "      root_paths=", list(robot.root_physx_view.prim_paths[: min(4, len(robot.root_physx_view.prim_paths))])
        )

    print("[inspect] sensors:")
    for name, sensor in env.scene.sensors.items():
        print("  ", name, type(sensor).__name__, "cfg_path=", sensor.cfg.prim_path)
        try:
            print("      instances=", sensor.num_instances)
        except Exception as exc:  # pragma: no cover - diagnostic output only
            print("      instances=<error>", repr(exc))

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
