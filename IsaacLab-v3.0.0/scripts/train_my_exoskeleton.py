"""外骨骼 PPO 训练的命令行入口。

这个文件是用户执行 ``python scripts/train_my_exoskeleton.py ...`` 时首先进入的
“适配层”，调用者通常是终端、Shell 脚本或 IDE 的运行配置。它本身不实现 PPO
算法，而是把本项目便于记忆的参数（例如 ``--task_id BT02``、``--imu 0``）转换为
Isaac Lab/Hydra 能识别的配置覆盖项，最后交给
``isaaclab_rl.entrypoints.run_train_cli`` 创建环境并训练策略。

输入来自 ``sys.argv``，输出是后端训练入口的进程退出码；checkpoint、TensorBoard
日志等文件由 RL-Games 后端按其配置写入 ``logs/``。启动顺序必须保持为：

1. 必要时用 Isaac Sim 自带的 USD 运行库重新启动当前 Python 进程；
2. 把本仓库的自定义包加入导入路径，并导入任务注册模块；
3. 校验并翻译外骨骼专用参数，移除被关闭的相机实体；
4. 补齐默认任务和 RL 后端，然后进入 Isaac Lab 的通用训练流程。

这里处理的是“启动参数”，不是机器人控制循环；修改本文件不会直接改变奖励函数、
观测向量或电机动作定义。
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

_KIT_USD_BOOTSTRAP_ENV = "_ISAACLAB_KIT_USD_BOOTSTRAPPED"

# 这些元组列出用户可输入的同义参数；对应的字符串常量是最终交给 Hydra 的字段路径。
_TASK_ID_OPTIONS = ("--task_id", "--task-id")
_DATASET_OPTIONS = ("--dataset", "--imitation_dataset", "--imitation-dataset")
_SENSOR_OPTIONS = {
    "--imu": "env.use_imu=",
    "--rgb": "env.use_rgb=",
    "--event": "env.use_event=",
    "--contact": "env.use_contact=",
}
_IMITATION_SENSOR_ENTITIES = {
    "rgb": ("left_rgb_camera", "right_rgb_camera"),
    "event": ("left_event_camera", "right_event_camera"),
}
_REFERENCE_PATH_OVERRIDE = "env.commands.base_velocity.reference_data_path="
_DATASET_PATH_OVERRIDE = "env.imitation_dataset_path="
_IMITATION_TASK_OVERRIDE = "env.imitation_task_id="
_COMMAND_TASK_OVERRIDE = "env.commands.base_velocity.task_id="
_IMITATION_TASK_IDS = frozenset(
    {"ALL", "NONE", "BT01", "BT02", "BT03", "BT04", "BT05", "AT01", "AT02", "AT03", "AT04", "AT05"}
)
_IMITATION_GYM_TASK = "Isaac-MyExoskeleton-Imitation-v0"


def _parse_sensor_bool(option: str, value: str) -> bool:
    """把命令行中的传感器开关转换为 Python 布尔值。

    在入口处转换可以让训练和回放给出一致、易懂的错误信息，也能阻止非法值进入
    Hydra/RL-Games。文档推荐 ``0``/``1``，同时接受 Shell 中常见的英文布尔写法。

    Args:
        option: 当前参数名，例如 ``--imu``；仅用于生成错误信息。
        value: 用户输入的原始字符串，首尾空白和大小写不影响解析。

    Returns:
        ``True`` 表示启用传感器，``False`` 表示禁用。

    Raises:
        ValueError: ``value`` 不是支持的布尔写法。
    """

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{option} expects one of 0, 1, true, false, yes, no, on, off; got {value!r}")


def _consume_sensor_cli_args(argv: list[str]) -> list[str]:
    """取出传感器快捷开关，并生成等价的环境配置覆盖项。

    快捷开关会在 Isaac Lab 启动器解析参数前被移除。显式输入会变成
    ``env.use_imu=False`` 之类的结构化配置；未输入的开关不会覆盖环境默认值。
    ``--imu=0`` 和 ``--imu 0`` 两种形式都可用，重复相同值无害，互相冲突的重复值
    会在 Kit 启动前失败。

    Args:
        argv: 不包含程序名的命令行参数列表。函数不会原地修改该列表。

    Returns:
        新参数列表；普通 Isaac Lab/Hydra 参数保持原顺序，传感器快捷开关被替换为
        ``env.use_*`` 覆盖项。

    Raises:
        ValueError: 开关缺少值、值无法解析，或同一开关被赋予冲突值。
    """

    remaining: list[str] = []
    selected: dict[str, bool] = {}
    index = 0
    while index < len(argv):
        argument = argv[index]
        option = argument.partition("=")[0]
        if option not in _SENSOR_OPTIONS:
            remaining.append(argument)
            index += 1
            continue

        if "=" in argument:
            value = argument.partition("=")[2]
        else:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"{option} requires a boolean value (0/1 or true/false)")
            value = argv[index + 1]
            index += 1
        if not value:
            raise ValueError(f"{option} requires a boolean value (0/1 or true/false)")

        enabled = _parse_sensor_bool(option, value)
        if option in selected and selected[option] != enabled:
            raise ValueError(f"conflicting {option} values: {selected[option]!r} and {enabled!r}")
        if option not in selected:
            remaining.append(f"{_SENSOR_OPTIONS[option]}{enabled}")
        selected[option] = enabled
        index += 1

    return remaining


def _apply_sensor_scene_overrides(argv: list[str]) -> list[str]:
    """在 Kit 扫描配置前，把已禁用的模仿相机实体设为 ``null``。

    ``launch_simulation`` 会先扫描完整环境配置，之后环境构造函数才有机会移除传感器。
    因此只设置 ``env.use_rgb=False`` 仍可能让默认 ``CameraCfg`` 请求 RTX 渲染。
    这里额外写入 ``env.scene.<camera>=null``，只阻止实体创建，不改变 PPO 固定长度的
    观测契约。

    Args:
        argv: 已包含 ``env.use_*`` 覆盖项的参数列表。

    Returns:
        非模仿任务原样返回；模仿任务则返回带相机 ``null`` 覆盖项的新列表。
    """

    task_name: str | None = None
    for index, argument in enumerate(argv):
        if argument.startswith("--task="):
            task_name = argument.partition("=")[2]
        elif argument == "--task" and index + 1 < len(argv):
            task_name = argv[index + 1]
    is_imitation = task_name == _IMITATION_GYM_TASK or any(
        argument.startswith((_IMITATION_TASK_OVERRIDE, _DATASET_PATH_OVERRIDE)) for argument in argv
    )
    if not is_imitation:
        return argv

    disabled: set[str] = set()
    false_values = {"0", "false", "no", "off"}
    for argument in argv:
        key, separator, value = argument.partition("=")
        if separator and key.startswith("env.use_") and value.strip().lower() in false_values:
            disabled.add(key.removeprefix("env.use_"))
    if not disabled:
        return argv

    result = list(argv)
    for sensor_name, entities in _IMITATION_SENSOR_ENTITIES.items():
        if sensor_name not in disabled:
            continue
        for entity in entities:
            prefix = f"env.scene.{entity}="
            # 直接覆盖场景时，不能重新启用已被 ``--rgb=0``/``--event=0`` 明确关闭的
            # 通道。这里原位替换参数，使 Hydra 对该字段只收到一个没有歧义的值。
            result = [argument for argument in result if not argument.startswith(prefix)]
            result.append(f"{prefix}null")
    return result


def _consume_exoskeleton_cli_args(argv: list[str]) -> list[str]:
    """把任务、数据集和传感器别名翻译成后端可识别的参数。

    RL 后端会把未知参数继续转发给 Hydra；若不在这里消费项目自定义参数，Hydra 会把
    它们当成格式错误的覆盖项。BT/AT 编号会选择模仿任务，普通 Gym ID 则保持旧版
    ``--task_id`` 用法。数据集路径会根据最终任务写入正确的配置命名空间。

    Args:
        argv: 不含程序名的原始命令行参数；普通后端参数和 Hydra 覆盖项也可混在其中。

    Returns:
        可直接传给 Isaac Lab 入口的全新参数列表。自定义别名会被移除，并补入
        ``--task``、任务编号、数据集路径和传感器配置覆盖项。

    Raises:
        ValueError: 参数缺值、重复选择互相冲突、BT/AT 编号与 Gym 任务不匹配，或
        数据集快捷参数和显式 Hydra 路径覆盖同时出现。
    """
    remaining: list[str] = []
    task_id: str | None = None
    dataset_path: str | None = None
    index = 0
    options = (*_TASK_ID_OPTIONS, *_DATASET_OPTIONS)
    while index < len(argv):
        argument = argv[index]
        option = argument.partition("=")[0]
        if option not in options:
            remaining.append(argument)
            index += 1
            continue

        if "=" in argument:
            value = argument.partition("=")[2]
        else:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"{option} requires a non-empty value")
            value = argv[index + 1]
            index += 1
        if not value:
            raise ValueError(f"{option} requires a non-empty value")

        if option in _TASK_ID_OPTIONS:
            if task_id is not None and task_id != value:
                raise ValueError(f"conflicting task selectors: {task_id!r} and {value!r}")
            task_id = value
        else:
            if dataset_path is not None and dataset_path != value:
                raise ValueError(f"conflicting dataset selectors: {dataset_path!r} and {value!r}")
            dataset_path = value
        index += 1

    explicit_tasks: list[str] = []
    index = 0
    while index < len(remaining):
        argument = remaining[index]
        if argument == "--task":
            if index + 1 >= len(remaining) or remaining[index + 1].startswith("--"):
                raise ValueError("--task requires a non-empty value")
            explicit_tasks.append(remaining[index + 1])
            index += 2
        elif argument.startswith("--task="):
            value = argument.partition("=")[2]
            if not value:
                raise ValueError("--task requires a non-empty value")
            explicit_tasks.append(value)
            index += 1
        else:
            index += 1
    if len(set(explicit_tasks)) > 1:
        raise ValueError(f"conflicting --task values: {explicit_tasks!r}")
    imitation_task: str | None = None
    if task_id is not None:
        normalized_task_id = task_id.upper()
        if normalized_task_id in _IMITATION_TASK_IDS:
            imitation_task = normalized_task_id
            if explicit_tasks and explicit_tasks[0] != _IMITATION_GYM_TASK:
                raise ValueError(
                    f"dataset task IDs require --task Isaac-MyExoskeleton-Imitation-v0; got {explicit_tasks[0]!r}"
                )
            remaining.extend(
                [
                    f"{_IMITATION_TASK_OVERRIDE}{imitation_task}",
                    f"{_COMMAND_TASK_OVERRIDE}{imitation_task}",
                ]
            )
            if not explicit_tasks:
                remaining.extend(["--task", _IMITATION_GYM_TASK])
        else:
            # 向后兼容：旧流程会把 --task_id 本身用作 Gym 环境 ID。
            if explicit_tasks and explicit_tasks[0] != task_id:
                raise ValueError(f"--task and --task_id select different tasks: {explicit_tasks[0]!r} and {task_id!r}")
            if not explicit_tasks:
                remaining.extend(["--task", task_id])

    # 把路径送到所选环境支持的配置命名空间。旧版平地配置只有
    # ``reference_data_path``，模仿配置则拥有 ``imitation_dataset_path``。
    selected_task = explicit_tasks[0] if explicit_tasks else None
    selected_imitation = selected_task == _IMITATION_GYM_TASK or imitation_task is not None
    reference_overrides = [arg for arg in remaining if arg.startswith(_REFERENCE_PATH_OVERRIDE)]
    dataset_overrides = [arg for arg in remaining if arg.startswith(_DATASET_PATH_OVERRIDE)]
    if dataset_path is not None:
        conflicting_overrides = dataset_overrides if selected_imitation else reference_overrides
        if conflicting_overrides:
            raise ValueError(
                "dataset selector conflicts with "
                f"{_REFERENCE_PATH_OVERRIDE.rstrip('=')} or "
                f"{_DATASET_PATH_OVERRIDE.rstrip('=')} override; provide only one"
            )
        # repr() 可让包含空格、引号或 ``=`` 的路径通过项目标量覆盖解析器时，仍保持为
        # 一个完整字符串值。
        if selected_imitation:
            remaining.append(f"{_DATASET_PATH_OVERRIDE}{dataset_path!r}")
        else:
            remaining.append(f"{_REFERENCE_PATH_OVERRIDE}{dataset_path!r}")
        if not explicit_tasks and task_id is None:
            remaining.extend(["--task", _IMITATION_GYM_TASK])
    # 统一由这个聚合函数消费自定义参数；即使调用者只认识旧版辅助函数，也能在参数
    # 交给 Hydra 前移除传感器开关。
    return _consume_sensor_cli_args(remaining)


def _bootstrap_isaacsim_usd() -> None:
    """确保后续导入使用 Isaac Sim 自带的 USD/PXR 运行库。

    函数必须在导入 Isaac Lab 之前调用。若找到 ``omni.usd.libs``，它会设置
    ``PYTHONPATH`` 和 ``LD_LIBRARY_PATH``，再用 ``os.execvpe`` 原位重启当前进程；
    环境变量哨兵可防止无限重启。找不到捆绑运行库时安静返回，沿用当前环境。

    Returns:
        无返回值。发生重启时当前进程映像会被替换，只有新进程会继续执行。
    """
    if os.environ.get(_KIT_USD_BOOTSTRAP_ENV) == "1":
        return

    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        return
    site_packages = Path(purelib)
    extscache = site_packages / "isaacsim" / "extscache"
    usd_candidates = sorted(
        path for path in extscache.glob("omni.usd.libs-*") if (path / "pxr").is_dir() and (path / "bin").is_dir()
    )
    if not usd_candidates:
        return

    usd_root = usd_candidates[0]
    env = os.environ.copy()
    env[_KIT_USD_BOOTSTRAP_ENV] = "1"
    env["PYTHONPATH"] = os.pathsep.join(item for item in (str(usd_root), env.get("PYTHONPATH", "")) if item)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        item for item in (str(usd_root / "bin"), env.get("LD_LIBRARY_PATH", "")) if item
    )
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def main() -> int:
    """完成任务注册和参数适配，再把训练交给 Isaac Lab。

    Returns:
        Isaac Lab/RL-Games 训练入口返回的进程退出码，通常 ``0`` 表示正常结束。

    Raises:
        ValueError: 外骨骼专用命令行参数不合法。
        Exception: Isaac Sim 初始化、环境构建或训练后端发生的异常会原样向上传播。
    """
    _bootstrap_isaacsim_usd()

    repo_root = Path(__file__).resolve().parents[1]
    for source_dir in (repo_root / "source" / "isaaclab_assets", repo_root / "source" / "isaaclab_tasks"):
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))

    from isaaclab_rl.entrypoints import run_train_cli

    import isaaclab_assets.my_exoskeleton  # noqa: F401

    argv = _consume_exoskeleton_cli_args(list(sys.argv[1:]))
    # 为替换了聚合参数处理器的调用者保留这一显式步骤；聚合处理器已移除传感器别名，
    # 因此重复调用不会改变结果。
    argv = _consume_sensor_cli_args(argv)
    # Isaac Lab 3.0 使用 ``--visualizer none`` 进行无界面运行；这里继续兼容任务文档和
    # 旧版本使用的 ``--headless`` 写法。
    headless = "--headless" in argv
    argv = [arg for arg in argv if arg != "--headless"]
    if headless and "--visualizer" not in argv and not any(arg.startswith("--visualizer=") for arg in argv):
        argv.extend(["--visualizer", "none"])
    if "--task" not in argv and not any(arg.startswith("--task=") for arg in argv):
        default_task = (
            _IMITATION_GYM_TASK
            if any(arg.startswith((_IMITATION_TASK_OVERRIDE, _DATASET_PATH_OVERRIDE)) for arg in argv)
            else "Isaac-MyExoskeleton-Flat-v0"
        )
        argv.extend(["--task", default_task])
    argv = _apply_sensor_scene_overrides(argv)
    if "--rl_library" not in argv and not any(arg.startswith("--rl_library=") for arg in argv):
        argv.extend(["--rl_library", "rl_games"])
    return run_train_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
