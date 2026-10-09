"""加载并回放外骨骼策略的命令行入口。

用户通过 ``python scripts/play_my_exoskeleton.py ...`` 调用本文件。它不负责训练，
也不直接驱动实体电机；它的职责是把外骨骼专用参数转换为 Isaac Lab/Hydra 配置，
注册 Gym 任务，再调用 ``isaaclab_rl.entrypoints.run_play_cli`` 在仿真中加载 checkpoint。

输入来自 ``sys.argv``，其中既可包含通用回放参数，也可包含传感器、诊断输出、
``ruleresult`` 轨迹输出和可点击 IMU 探针等项目参数。输出是通用回放入口的退出码；
诊断 JSONL、关节轨迹 JSONL 或 55AA 二进制文件由环境按转换后的配置写出。

启动顺序为：准备 Isaac Sim 的 USD 运行库 -> 导入并注册外骨骼任务 -> 逐组消费项目
参数 -> 校验 IMU 可视化条件 -> 补齐默认任务/后端 -> 启动回放。参数必须在进入
RL-Games 前消费完，否则未知的 ``--...`` 可能被 Hydra 错当成配置表达式。
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

_KIT_USD_BOOTSTRAP_ENV = "_ISAACLAB_KIT_USD_BOOTSTRAPPED"

# 六个名称对应场景配置中的六个肢段 IMU，而不是实体机电机编号。
_IMU_SENSOR_NAMES = (
    "imu_01",
    "imu_02",
    "imu_03",
    "imu_04",
    "imu_05",
    "imu_06",
)

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
_DIAGNOSTICS_OPTIONS = ("--diagnostics", "--show-diagnostics")
_DIAGNOSTICS_INTERVAL_OPTIONS = ("--diagnostics_interval", "--diagnostics-interval")
_DIAGNOSTICS_OUTPUT_OPTIONS = (
    "--diagnostics_output",
    "--diagnostics-output",
    "--diagnostics-file",
)
_RULERESULT_OPTIONS = ("--ruleresult", "--write-ruleresult")
_RULERESULT_INTERVAL_OPTIONS = ("--ruleresult_interval", "--ruleresult-interval")
_RULERESULT_ENV_OPTIONS = ("--ruleresult_env_id", "--ruleresult-env-id")
_RULERESULT_OUTPUT_OPTIONS = (
    "--ruleresult_output",
    "--ruleresult-output",
    "--ruleresult-file",
)
_RULERESULT_TIME_MODE_OPTIONS = ("--ruleresult_time_mode", "--ruleresult-time-mode")
_RULERESULT_ACTUAL_TIME_OPTIONS = ("--ruleresult-use-actual-dt",)
_RULERESULT_FORMAT_OPTIONS = ("--ruleresult_format", "--ruleresult-format")
_RULERESULT_PROTOCOL_CONFIG_OPTIONS = (
    "--ruleresult_protocol_config",
    "--ruleresult-protocol-config",
)
_RULERESULT_PREDICTION_HORIZON_OPTIONS = (
    "--ruleresult_prediction_horizon",
    "--ruleresult-prediction-horizon",
)
_RULERESULT_PREDICTION_MODE_OPTIONS = (
    "--ruleresult_prediction_mode",
    "--ruleresult-prediction-mode",
)
_RULERESULT_PREDICTION_DT_OPTIONS = (
    "--ruleresult_prediction_dt",
    "--ruleresult-prediction-dt",
)
_REFERENCE_PATH_OVERRIDE = "env.commands.base_velocity.reference_data_path="
_DATASET_PATH_OVERRIDE = "env.imitation_dataset_path="
_IMITATION_TASK_OVERRIDE = "env.imitation_task_id="
_COMMAND_TASK_OVERRIDE = "env.commands.base_velocity.task_id="
_IMITATION_TASK_IDS = frozenset(
    {"ALL", "NONE", "BT01", "BT02", "BT03", "BT04", "BT05", "AT01", "AT02", "AT03", "AT04", "AT05"}
)
_IMITATION_GYM_TASK = "Isaac-MyExoskeleton-Imitation-v0"


def _parse_sensor_bool(option: str, value: str) -> bool:
    """把传感器开关的文本值转换成布尔值。

    Args:
        option: 参数名，例如 ``--rgb``，仅用于错误提示。
        value: 用户输入的原始值；允许 ``0/1`` 及常见英文布尔写法。

    Returns:
        是否启用该传感器。

    Raises:
        ValueError: 输入不属于支持的布尔值集合。
    """

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{option} expects one of 0, 1, true, false, yes, no, on, off; got {value!r}")


def _consume_sensor_cli_args(argv: list[str]) -> list[str]:
    """消费回放传感器开关，并生成环境配置覆盖项。

    支持 ``--imu=0`` 和 ``--imu 0``，RGB、事件相机和接触传感器同理。没有出现的
    开关不会覆盖环境默认值。函数返回新列表，不会原地修改调用者传入的 ``argv``。

    Args:
        argv: 不包含程序名的命令行参数列表。

    Returns:
        已把快捷开关替换为 ``env.use_*=True/False`` 的参数列表。

    Raises:
        ValueError: 开关缺值、值非法，或同一开关出现互相冲突的重复值。
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
    """在 Kit 扫描模仿任务配置前移除明确禁用的相机实体。

    启动器读取场景配置的时机早于环境构造函数移除可选传感器的时机，所以仅设置
    ``env.use_rgb=False`` 仍可能初始化 RTX 相机。显式的 ``null`` 覆盖会阻止相机
    创建，但固定 shape 的策略观测接口保持不变。

    Args:
        argv: 已完成传感器快捷开关转换的参数列表。

    Returns:
        非模仿任务保持原样；模仿任务返回附加 ``env.scene.*=null`` 的新列表。
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

    false_values = {"0", "false", "no", "off"}
    disabled = {
        key.removeprefix("env.use_")
        for argument in argv
        for key, separator, value in (argument.partition("="),)
        if separator and key.startswith("env.use_") and value.strip().lower() in false_values
    }
    if not disabled:
        return argv

    result = list(argv)
    for sensor_name, entities in _IMITATION_SENSOR_ENTITIES.items():
        if sensor_name not in disabled:
            continue
        for entity in entities:
            prefix = f"env.scene.{entity}="
            # 即使调用者还直接覆盖了该场景实体，也必须以显式的传感器关闭开关为准。
            result = [argument for argument in result if not argument.startswith(prefix)]
            result.append(f"{prefix}null")
    return result


def _consume_exoskeleton_cli_args(argv: list[str]) -> list[str]:
    """把任务、数据集和传感器别名转换为后端安全参数。

    ``BT02`` 之类的编号代表模仿数据子任务，而 ``Isaac-...-v0`` 代表 Gym 环境。
    该函数兼容两种历史用法，并将数据集路径送到所选环境真正拥有的配置字段。

    Args:
        argv: 不含程序名的原始参数列表。

    Returns:
        可继续传给诊断、输出和 Isaac Lab 参数处理阶段的新列表。

    Raises:
        ValueError: 参数缺值、任务选择冲突、数据任务与 Gym 环境不匹配，或数据集
        路径同时通过快捷参数和 Hydra 覆盖指定。
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
            if explicit_tasks and explicit_tasks[0] != task_id:
                raise ValueError(f"--task and --task_id select different tasks: {explicit_tasks[0]!r} and {task_id!r}")
            if not explicit_tasks:
                remaining.extend(["--task", task_id])

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
        if selected_imitation:
            remaining.append(f"{_DATASET_PATH_OVERRIDE}{dataset_path!r}")
        else:
            remaining.append(f"{_REFERENCE_PATH_OVERRIDE}{dataset_path!r}")
        if not explicit_tasks and task_id is None:
            remaining.extend(["--task", _IMITATION_GYM_TASK])
    # 统一由这个聚合函数消费自定义参数；即使调用者只认识旧版辅助函数，也能在参数
    # 交给 Hydra 前移除传感器开关。
    return _consume_sensor_cli_args(remaining)


def _consume_diagnostics_args(argv: list[str]) -> list[str]:
    """把回放诊断参数翻译为模仿环境配置。

    通用 ``--visualizer viser`` 和 Kit 参数仍由 Isaac Lab 负责；这里只消费项目专用
    的诊断开关、采样间隔和输出路径。间隔的单位是仿真控制步，不是秒。

    Args:
        argv: 已完成任务和传感器别名转换的参数列表。

    Returns:
        诊断选项被替换为 ``env.diagnostics_*`` 覆盖项的新列表。

    Raises:
        ValueError: 选项缺值，或诊断间隔不是正整数。
    """

    remaining: list[str] = []
    enabled = False
    interval: str | None = None
    output_path: str | None = None
    value_options = (*_DIAGNOSTICS_INTERVAL_OPTIONS, *_DIAGNOSTICS_OUTPUT_OPTIONS)
    index = 0
    while index < len(argv):
        argument = argv[index]
        option = argument.partition("=")[0]
        if option in _DIAGNOSTICS_OPTIONS:
            enabled = True
            index += 1
            continue
        if option not in value_options:
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
        if option in _DIAGNOSTICS_INTERVAL_OPTIONS:
            try:
                if int(value) <= 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(f"{option} must be a positive integer") from exc
            interval = value
        else:
            output_path = value
        enabled = True
        index += 1

    if enabled:
        remaining.append("env.diagnostics_enabled=True")
    if interval is not None:
        remaining.append(f"env.diagnostics_interval={interval}")
    if output_path is not None:
        remaining.append(f"env.diagnostics_output_path={output_path!r}")
    return remaining


def _consume_ruleresult_args(argv: list[str]) -> list[str]:  # noqa: C901
    """把关节结果输出参数转换为 ``ruleresult`` 环境配置。

    该函数只准备配置，不在这里打开文件。环境运行后可按指定周期输出 JSONL，或按
    审核过的协议配置输出 55AA 二进制帧；预测步数和预测间隔也会一并传给记录器。
    相对输出路径在此处绑定到启动目录，避免后端切换运行目录后文件出现在意外位置。

    Args:
        argv: 仍包含通用回放参数和可选 ``--ruleresult-*`` 参数的列表。

    Returns:
        删除所有 ``ruleresult`` 快捷参数并加入 ``env.ruleresult_*`` 覆盖项的新列表。

    Raises:
        ValueError: 数值范围、时间模式、预测模式或输出格式不合法，或必需值缺失。
    """

    remaining: list[str] = []
    enabled = False
    interval: str | None = None
    env_id: str | None = None
    output_path: str | None = None
    time_mode: str | None = None
    output_format: str | None = None
    protocol_config_path: str | None = None
    prediction_horizon: str | None = None
    prediction_mode: str | None = None
    prediction_dt: str | None = None
    value_options = (
        *_RULERESULT_INTERVAL_OPTIONS,
        *_RULERESULT_ENV_OPTIONS,
        *_RULERESULT_OUTPUT_OPTIONS,
        *_RULERESULT_TIME_MODE_OPTIONS,
        *_RULERESULT_FORMAT_OPTIONS,
        *_RULERESULT_PROTOCOL_CONFIG_OPTIONS,
        *_RULERESULT_PREDICTION_HORIZON_OPTIONS,
        *_RULERESULT_PREDICTION_MODE_OPTIONS,
        *_RULERESULT_PREDICTION_DT_OPTIONS,
    )
    index = 0
    while index < len(argv):
        argument = argv[index]
        option = argument.partition("=")[0]
        if option in _RULERESULT_OPTIONS:
            enabled = True
            index += 1
            continue
        if option in _RULERESULT_ACTUAL_TIME_OPTIONS:
            enabled = True
            time_mode = "wall_clock"
            index += 1
            continue
        if option not in value_options:
            remaining.append(argument)
            index += 1
            continue
        if "=" in argument:
            value = argument.partition("=")[2]
        else:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"{option} requires a value")
            value = argv[index + 1]
            index += 1
        if not value:
            raise ValueError(f"{option} requires a non-empty value")
        if option in _RULERESULT_INTERVAL_OPTIONS:
            try:
                if int(value) <= 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(f"{option} must be a positive integer") from exc
            interval = value
        elif option in _RULERESULT_ENV_OPTIONS:
            try:
                if int(value) < 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(f"{option} must be a non-negative integer") from exc
            env_id = value
        elif option in _RULERESULT_TIME_MODE_OPTIONS:
            normalized = value.strip().lower()
            aliases = {"sim": "simulation", "wall": "wall_clock", "actual": "wall_clock"}
            normalized = aliases.get(normalized, normalized)
            if normalized not in {"simulation", "wall_clock", "configured"}:
                raise ValueError(f"{option} must be simulation, wall_clock, or configured")
            time_mode = normalized
        elif option in _RULERESULT_FORMAT_OPTIONS:
            normalized = value.strip().lower()
            normalized = {"json": "jsonl", "55aa": "protocol_55aa", "protocol-55aa": "protocol_55aa"}.get(
                normalized, normalized
            )
            if normalized not in {"jsonl", "protocol_55aa"}:
                raise ValueError(f"{option} must be jsonl or protocol_55aa")
            output_format = normalized
        elif option in _RULERESULT_PROTOCOL_CONFIG_OPTIONS:
            resolved_config = Path(value).expanduser()
            if not resolved_config.is_absolute():
                resolved_config = (Path.cwd() / resolved_config).resolve()
            protocol_config_path = str(resolved_config)
        elif option in _RULERESULT_PREDICTION_HORIZON_OPTIONS:
            try:
                if int(value) < 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(f"{option} must be a non-negative integer") from exc
            prediction_horizon = value
        elif option in _RULERESULT_PREDICTION_MODE_OPTIONS:
            normalized = value.strip().lower()
            if normalized in {"constant_acceleration", "extrapolate"}:
                normalized = "kinematic"
            if normalized not in {"none", "kinematic", "supplied"}:
                raise ValueError(f"{option} must be none, kinematic, or supplied")
            prediction_mode = normalized
        else:
            if option in _RULERESULT_PREDICTION_DT_OPTIONS:
                try:
                    if float(value) <= 0.0:
                        raise ValueError
                except ValueError as exc:
                    raise ValueError(f"{option} must be a positive number") from exc
                prediction_dt = value
            else:
                # 先以启动目录为基准解析命令行路径，再让 RL 后端把 ``env.cfg.log_dir``
                # 指向 checkpoint 运行目录。这样 ``--ruleresult-output logs/foo.jsonl``
                # 就不会悄悄变成 ``<run_dir>/logs/foo.jsonl``。
                resolved_output = Path(value).expanduser()
                if not resolved_output.is_absolute():
                    resolved_output = (Path.cwd() / resolved_output).resolve()
                output_path = str(resolved_output)
        enabled = True
        index += 1

    if enabled:
        remaining.append("env.ruleresult_enabled=True")
    if interval is not None:
        remaining.append(f"env.ruleresult_interval={interval}")
    if env_id is not None:
        remaining.append(f"env.ruleresult_env_id={env_id}")
    if output_path is not None:
        remaining.append(f"env.ruleresult_output_path={output_path!r}")
    if time_mode is not None:
        remaining.append(f"env.ruleresult_time_mode={time_mode}")
    if protocol_config_path is not None and output_format is None:
        output_format = "protocol_55aa"
    if output_format is not None:
        remaining.append(f"env.ruleresult_output_format={output_format}")
    if protocol_config_path is not None:
        remaining.append(f"env.ruleresult_protocol_config_path={protocol_config_path!r}")
    if prediction_horizon is not None:
        remaining.append(f"env.ruleresult_prediction_horizon={prediction_horizon}")
    if prediction_mode is not None:
        remaining.append(f"env.ruleresult_prediction_mode={prediction_mode}")
    if prediction_dt is not None:
        remaining.append(f"env.ruleresult_prediction_dt_s={prediction_dt}")
    return remaining


def _bootstrap_isaacsim_usd() -> None:
    """在导入 Isaac Lab 前准备 Isaac Sim 捆绑的 USD/PXR 运行库。

    找到 ``omni.usd.libs`` 后，函数通过环境变量加入 Python 模块和动态库路径，并用
    ``os.execvpe`` 原位重启当前进程。哨兵变量保证只重启一次；找不到运行库则沿用
    当前环境并直接返回。

    Returns:
        无返回值；需要重启时，当前进程映像会被新的 Python 进程替换。
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


def _enable_imu_probes(argv: list[str]) -> list[str]:
    """把 ``--show-imu`` 转换为六个 IMU 的可视化配置。

    可点击探针只适合单环境 Kit 视口：多环境会让探针与读数难以一一对应，Viser
    等其他可视化器也不能显示 Kit 的 USD 选择属性。

    Args:
        argv: 回放参数列表。

    Returns:
        移除 ``--show-imu`` 后的参数；启用时额外包含六个 ``debug_vis=True`` 覆盖项。

    Raises:
        ValueError: 启用探针但没有显式设置 ``--num_envs=1`` 或没有选择 Kit。
    """
    show_imu = "--show-imu" in argv
    argv = [arg for arg in argv if arg != "--show-imu"]
    if not show_imu:
        return argv

    if not _uses_single_environment(argv):
        raise ValueError("--show-imu requires --num_envs=1 so every clickable probe has one sensor reading.")
    if not _uses_kit_visualizer(argv):
        raise ValueError("--show-imu requires --visualizer kit so the selectable probes are visible.")

    return argv + [f"env.scene.{sensor_name}.debug_vis=True" for sensor_name in _IMU_SENSOR_NAMES]


def _uses_single_environment(argv: list[str]) -> bool:
    """检查参数是否显式请求一个环境。

    Args:
        argv: 命令行参数列表，支持 ``--num_envs=1`` 和 ``--num_envs 1``。

    Returns:
        仅在参数明确写出环境数为 ``1`` 时返回 ``True``；省略参数也返回 ``False``。
    """
    for index, arg in enumerate(argv):
        if arg.startswith("--num_envs="):
            return arg.partition("=")[2] == "1"
        if arg == "--num_envs" and index + 1 < len(argv):
            return argv[index + 1] == "1"
    return False


def _uses_kit_visualizer(argv: list[str]) -> bool:
    """检查逗号分隔的可视化器列表中是否包含 Kit。

    Args:
        argv: 命令行参数列表。

    Returns:
        ``--visualizer`` 的值中包含独立的 ``kit`` 项时返回 ``True``。
    """
    for index, arg in enumerate(argv):
        value = None
        if arg.startswith("--visualizer="):
            value = arg.partition("=")[2]
        elif arg == "--visualizer" and index + 1 < len(argv):
            value = argv[index + 1]
        if value is not None and "kit" in {item.strip() for item in value.split(",")}:
            return True
    return False


def main() -> int:
    """注册外骨骼任务、完成全部参数适配并启动策略回放。

    Returns:
        Isaac Lab/RL-Games 回放入口返回的进程退出码。

    Raises:
        ValueError: 任一项目专用参数不合法或 IMU 探针启动条件不满足。
        Exception: checkpoint 加载、仿真初始化或环境运行异常会原样向上传播。
    """
    _bootstrap_isaacsim_usd()

    repo_root = Path(__file__).resolve().parents[1]
    for source_dir in (repo_root / "source" / "isaaclab_assets", repo_root / "source" / "isaaclab_tasks"):
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))

    from isaaclab_rl.entrypoints import run_play_cli

    import isaaclab_assets.my_exoskeleton  # noqa: F401

    argv = _consume_exoskeleton_cli_args(list(sys.argv[1:]))
    # 为替换了聚合参数处理器的调用者保留这一显式步骤；聚合处理器已移除传感器别名，
    # 因此重复调用不会改变结果。
    argv = _consume_sensor_cli_args(argv)
    argv = _consume_diagnostics_args(argv)
    argv = _consume_ruleresult_args(argv)
    argv = _enable_imu_probes(argv)
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
    return run_play_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
