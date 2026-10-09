# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""验证并以空运行方式回放外骨骼预测轨迹。

这是部署工具的命令行入口，通常由操作者直接运行。输入是 ``prediction.json``
或 JSONL 轨迹文件，以及可选的关节到电机标定文件；输出是校验结果和逐周期的
模拟电机命令。当前入口只支持空运行（dry-run，即不连接实体硬件），因此不会打开
CAN、EtherCAT、串口或 ROS 2 总线。

一次回放按以下顺序调用部署模块：

1. :mod:`deployment.trajectory` 读取轨迹并按控制时刻插值；
2. :mod:`deployment.joint_mapping` 将仿真关节量换算为电机侧物理量；
3. :mod:`deployment.safety` 检查位置、速度、加速度、反馈与时序；
4. :mod:`deployment.replay` 按统一周期生成完整的八电机命令；
5. :class:`deployment.motor_bus.DryRunBus` 模拟发送和反馈。

建议第一次阅读部署代码时先看本文件，再按上面的模块顺序继续阅读。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # Direct invocation: ``python scripts/deploy_prediction.py``.
    from deployment.joint_mapping import JointMapper, MappingError
    from deployment.motor_bus import DryRunBus
    from deployment.replay import ReplayConfig, replay
    from deployment.safety import SafetyConfig, SafetyLimiter, SafetyViolation
    from deployment.trajectory import TrajectoryError, load_trajectory
except ImportError:  # Module invocation: ``python -m scripts.deploy_prediction``.
    from scripts.deployment.joint_mapping import JointMapper, MappingError
    from scripts.deployment.motor_bus import DryRunBus
    from scripts.deployment.replay import ReplayConfig, replay
    from scripts.deployment.safety import SafetyConfig, SafetyLimiter, SafetyViolation
    from scripts.deployment.trajectory import TrajectoryError, load_trajectory


def _parser() -> argparse.ArgumentParser:
    """创建本命令的参数解析器。

    Returns:
        已注册轨迹路径、插值方式、控制周期和空运行选项的解析器。这里只定义
        命令行格式，不读取文件，也不会启动回放。
    """

    parser = argparse.ArgumentParser(
        description="Validate/replay a JSON or JSONL trajectory (dry-run only)."
    )
    parser.add_argument("prediction", type=Path, help="trajectory file")
    parser.add_argument("--mapping", type=Path, help="JSON joint-to-motor calibration")
    parser.add_argument("--episode", type=int, help="episode to select from an appended JSONL file")
    parser.add_argument(
        "--use-nested-prediction",
        action="store_true",
        help="use ruleresult prediction horizons instead of top-level samples",
    )
    parser.add_argument("--period", type=float, default=0.02, help="hardware command period in seconds")
    parser.add_argument("--interpolation", choices=("linear", "hermite3", "quintic"), default="linear")
    parser.add_argument("--mode", choices=("position", "position_velocity_ff"), default="position")
    parser.add_argument("--max-lateness", type=float, default=0.05)
    parser.add_argument("--fast", action="store_true", help="do not wait for wall time (CI/dry-run)")
    parser.add_argument("--check-only", action="store_true", help="validate and print metadata without replay")
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="reserved; always rejected until a reviewed adapter exists",
    )
    parser.add_argument("--quiet", action="store_true", help="do not print every dry-run frame")
    return parser


def main(argv: list[str] | None = None) -> int:
    """校验命令行输入，并在空运行总线上执行完整部署流程。

    Args:
        argv: 待解析的命令行参数列表。不传时由 :mod:`argparse` 读取当前进程的
            ``sys.argv``，测试代码可以传入列表以避免依赖真实终端。

    Returns:
        进程退出码：``0`` 表示校验或回放成功，``1`` 表示回放期间触发安全停止，
        ``2`` 表示输入/配置无效，``130`` 表示用户用 ``Ctrl+C`` 中断。

    Notes:
        本函数构造的始终是 :class:`DryRunBus`。即使传入 ``--no-dry-run`` 也会
        明确拒绝执行，防止尚未审查的通用示例意外驱动实体电机。
    """

    args = _parser().parse_args(argv)
    if args.no_dry_run:
        print(
            "拒绝启动：仓库尚未接入经过审查的真实硬件适配器；当前入口只能 dry-run。",
            file=sys.stderr,
        )
        return 2
    try:
        trajectory = load_trajectory(
            args.prediction,
            episode=args.episode,
            use_nested_prediction=args.use_nested_prediction,
        )
        mapper = (
            JointMapper.from_file(args.mapping, trajectory.joint_names)
            if args.mapping is not None
            else JointMapper.identity(trajectory.joint_names)
        )
        safety = SafetyLimiter(mapper.hardware_ids, SafetyConfig())
    except (OSError, ValueError, TrajectoryError, MappingError) as exc:
        print(f"轨迹或映射校验失败：{exc}", file=sys.stderr)
        return 2

    print(
        f"validated {len(trajectory.samples)} samples, {trajectory.joint_count} joints, "
        f"duration={trajectory.duration_s:.3f}s, period={args.period:.4f}s, "
        f"interpolation={args.interpolation}, mode={args.mode}"
    )
    if args.check_only:
        return 0

    bus = DryRunBus(mapper.hardware_ids)
    callback = None if args.quiet else lambda frame: print(
        f"dry-run seq={frame.sequence} t_ns={frame.timestamp_ns} q0={frame.position_rad[0]:.6f}"
    )
    try:
        count = replay(
            trajectory,
            bus,
            mapper,
            safety,
            ReplayConfig(
                control_period_s=args.period,
                interpolation=args.interpolation,
                mode=args.mode,
                max_lateness_s=args.max_lateness,
                realtime=not args.fast,
            ),
            on_command=callback,
        )
    except KeyboardInterrupt:
        print("收到中断，已执行安全停机。", file=sys.stderr)
        return 130
    except (SafetyViolation, RuntimeError, ValueError) as exc:
        print(f"回放已停止：{exc}", file=sys.stderr)
        return 1
    print(f"dry-run complete: {count} atomic command cycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
