"""外骨骼基础行走任务的运行时逻辑。

新手可以把本文件理解成环境的“计算公式集合”，它和环境配置文件配合使用：配置文件
决定启用哪些项目及其权重，本文件负责真正计算观测、步态命令、奖励和课程阶段。
Isaac Lab 会同时处理许多个并行环境，因此绝大多数张量的第 0 维都是 num_envs。

推荐阅读顺序：

1. MyExoskeletonFlatEnv：在 Isaac Lab 的标准 step 后记录输出；
2. GaitVelocityCommand：维护五种步态、相位和速度命令；
3. exoskeleton_observation：拼接 actor 可以看到的输入；
4. exoskeleton_critic_observation：拼接仅训练期使用的特权输入；
5. 本文件后半部分的奖励函数：每个函数返回 shape [num_envs]；
6. advance_exoskeleton_curriculum：按全局控制步开放更难的任务。

这里的关节角单位为 rad，角速度为 rad/s，力为 N，线速度为 m/s。代码中的网络动作
仍是 8 维归一化值，不是实体电机协议中的原始整数。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_assets.tasks.gait import (
    FROUDE_THRESHOLD,
    GAIT_MODE_COUNT,
    GRAVITY_M_S2,
    LEG_LENGTH_M,
    CurriculumPhase,
    GaitMode,
    curriculum_phase_for_step,
)
from isaaclab_assets.tasks.gait import (
    gait_mask as _shared_gait_mask,
)
from isaaclab_assets.tasks.robot_contract import (
    ACTUATED_JOINT_NAMES,
    IMU_SENSOR_NAMES,
    KNEE_ARTICULATION_LIMITS_RAD,
    KNEE_ARTICULATION_SIGNS,
    KNEE_JOINT_NAMES,
)
from isaaclab_assets.tasks.ruleresult import RuleresultRecorderMixin
from isaaclab_assets.tasks.sensor_reference import SensorReferenceBuffer

if TYPE_CHECKING:
    from isaaclab_assets.envs.my_exoskeleton_env_cfg import MyExoskeletonFlatEnvCfg


# 足底法向力小于该阈值时按传感器噪声处理，而不是一次真实着地。
CONTACT_FORCE_THRESHOLD_N = 5.0
GAIT_UPDATE_PERIOD_S = 0.1
"""步态命令层更新周期 [s]，即论文中的 10 Hz；策略控制本身仍为 50 Hz。"""

WALK_TO_STAND_HOLD_S = 1.5
RUN_TO_WALK_HOLD_S = 2.5


# -----------------------------------------------------------------------------
# 张量兼容与规范关节顺序
# -----------------------------------------------------------------------------


def _optional_sensor_enabled(env, name: str) -> bool:
    """读取可选传感器开关，并兼容没有该字段的旧配置。

    Args:
        env: Isaac Lab 环境或测试替身。
        name: 传感器通道短名称，例如 ``imu``、``contact``。

    Returns:
        配置中的 ``use_<name>`` 布尔值；旧配置没有该字段时返回 ``True``。
    """

    cfg = getattr(env, "cfg", None)
    return bool(getattr(cfg, f"use_{name}", True))


def _as_tensor(value: torch.Tensor) -> torch.Tensor:
    """统一取得原生张量或仿真包装对象暴露的 Torch 视图。

    Args:
        value: :class:`torch.Tensor`，或带有 ``.torch`` 属性的 Isaac Lab 数组。

    Returns:
        与输入共享同一份仿真数据的 Torch 张量；不复制数值。
    """

    return value.torch if hasattr(value, "torch") else value


def _canonical_joint_tensor(env, value: torch.Tensor) -> torch.Tensor:
    """把关节张量最后一维重排为项目公开的八关节固定顺序。

    Args:
        env: 包含 ``scene["robot"]`` 的环境。
        value: 关节数据，通常 shape 为 ``[num_envs, 8]``；最后一维不是八时原样
            返回，供不适用关节重排的测试数据继续使用。

    Returns:
        shape 与输入相同、最后一维按 ``ACTUATED_JOINT_NAMES`` 排列的张量。首次
        调用解析名称下标，之后缓存在环境上，避免每个控制周期重复查找。

    Raises:
        ValueError: 机器人缺少任一规范关节。
    """

    value = _as_tensor(value)
    if value.ndim < 2 or value.shape[-1] != len(ACTUATED_JOINT_NAMES):
        return value
    robot = env.scene["robot"]
    names = tuple(getattr(robot, "joint_names", ()))
    if not names:
        return value
    indices = getattr(env, "_my_exoskeleton_joint_ids", None)
    if indices is None:
        missing = [name for name in ACTUATED_JOINT_NAMES if name not in names]
        if missing:
            raise ValueError(f"robot is missing canonical joints: {missing}")
        indices = torch.tensor([names.index(name) for name in ACTUATED_JOINT_NAMES], device=value.device)
        setattr(env, "_my_exoskeleton_joint_ids", indices)
    return value.index_select(-1, indices)


# -----------------------------------------------------------------------------
# 环境与五步态命令状态机
# -----------------------------------------------------------------------------


class MyExoskeletonFlatEnv(RuleresultRecorderMixin, ManagerBasedRLEnv):
    """八自由度外骨骼的 Isaac Lab 管理器式强化学习环境。

    大部分训练逻辑由 Isaac Lab 的动作、观测、奖励和终止管理器执行。本子类只在
    标准 ``step`` 结束后调用 :class:`RuleresultRecorderMixin`，把策略目标整理成
    可选的结构化输出。
    """

    cfg: MyExoskeletonFlatEnvCfg

    def step(self, action: torch.Tensor):
        """推进一个策略控制周期，并按需记录结构化模型结果。

        Args:
            action: Actor 输出的归一化动作，shape 为 ``[num_envs, 8]``。它不是
                电机角度，也不是 55AA 协议整数。

        Returns:
            与 :meth:`ManagerBasedRLEnv.step` 完全相同的环境步进结果。
        """

        result = super().step(action)
        self._ruleresult_record_after_step(result)
        return result


class GaitVelocityCommand(UniformVelocityCommand):
    """带五种步态有限状态机的速度命令项。

    Isaac Lab 原始命令项每隔数秒重新采样一次目标速度，本状态机则以 10 Hz 检查
    Walk、Stand、Walk-to-Stand、Run 和 Run-to-Walk 的切换条件。分开这两个频率，
    既能让一个目标保持足够长时间，又能累计停稳 1.5 s、跑转走 2.5 s 等条件。

    所有逐环境状态张量的 shape 均为 ``[num_envs]``：``gait_mode`` 是整数标签，
    ``gait_phase`` 是 ``[0, 2*pi)`` 的相位 [rad]，``transition_time`` 和
    ``stable_time`` 的单位为秒。
    """

    def __init__(self, cfg, env):
        """创建速度命令项及每个并行环境的步态状态。

        Args:
            cfg: ``UniformVelocityCommand`` 配置，包含速度范围、参考数据路径和
                重采样周期。
            env: 拥有 ``num_envs`` 个并行实例的 Isaac Lab 环境。

        Notes:
            参考路径是目录时留给多模态模仿环境加载；是文件时由旧版
            :class:`SensorReferenceBuffer` 直接读取。这样两种数据路径不会重复加载。
        """

        super().__init__(cfg, env)
        # 模仿环境可按数据集任务固定每个环境的前进速度；NaN 表示不覆盖，让普通
        # 平地任务继续使用课程/配置中的随机命令范围。
        self.task_speed_m_s = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.gait_mode = torch.full((self.num_envs,), int(GaitMode.WALK), dtype=torch.long, device=self.device)
        self.gait_phase = torch.rand(self.num_envs, device=self.device) * (2.0 * math.pi)
        self.transition_time = torch.zeros(self.num_envs, device=self.device)
        self.stable_time = torch.zeros(self.num_envs, device=self.device)
        self.previous_speed = torch.zeros(self.num_envs, device=self.device)
        self.curriculum_phase = CurriculumPhase.WALK_ONLY
        self._gait_update_accumulator = 0.0
        reference_path = getattr(cfg, "reference_data_path", None)
        self.reference_buffer = None
        self.reference_random_start = bool(getattr(cfg, "reference_random_start", False))
        # 目录表示多模态数据集根目录，由 MyExoskeletonImitationEnv 加载；旧版
        # SensorReferenceBuffer 只接受单个轨迹文件。
        self.imitation_dataset_path = None
        if reference_path and Path(reference_path).is_dir():
            self.imitation_dataset_path = str(reference_path)
        elif reference_path:
            self.reference_buffer = SensorReferenceBuffer.from_file(
                reference_path,
                device=self.device,
                target_dt_s=float(getattr(cfg, "reference_target_dt_s", 0.02)),
                loop=bool(getattr(cfg, "reference_loop", True)),
            )

    def _static_task_mask(self) -> torch.Tensor:
        """找出被数据集任务固定为静止目标的环境。

        ``task_speed_m_s`` 中的 NaN 表示普通平地任务仍使用命令采样器，不属于静止
        覆盖。只有“有限数值且速度绝对值小于 0.1 m/s”的任务（例如 BT01）才为静止。

        Returns:
            布尔掩码，shape 为 ``[num_envs]``；``True`` 表示必须保持 Stand。
        """

        return torch.isfinite(self.task_speed_m_s) & (self.task_speed_m_s.abs() < 0.1)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """重置选中环境的命令、相位、计时器和初始步态。

        Args:
            env_ids: 要重置的环境下标；``None`` 表示全部环境。

        Returns:
            父类命令项返回的重置指标字典。

        Notes:
            初始模式受当前课程阶段、命令速度和静止任务覆盖共同决定。多步态阶段
            使用 Froude 数区分 Walk 与 Run，而静止数据任务始终优先设为 Stand。
        """

        metrics = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.transition_time[ids] = 0.0
        self.stable_time[ids] = 0.0
        self.gait_phase[ids] = torch.rand_like(self.gait_phase[ids]) * (2.0 * math.pi)
        if self.reference_buffer is not None:
            self.reference_buffer.reset(ids, self.num_envs, random_start=self.reference_random_start)

        command_speed = torch.linalg.vector_norm(self.command[ids, :2], dim=-1)
        static = self._static_task_mask()
        if self.curriculum_phase == CurriculumPhase.WALK_ONLY:
            self.gait_mode[ids] = torch.where(
                static[ids],
                torch.full_like(self.gait_mode[ids], int(GaitMode.STAND)),
                torch.full_like(self.gait_mode[ids], int(GaitMode.WALK)),
            )
        else:
            initial_mode = torch.full_like(command_speed, int(GaitMode.WALK), dtype=torch.long)
            initial_mode[command_speed < 0.1] = int(GaitMode.WALK_TO_STAND)
            if self.curriculum_phase == CurriculumPhase.MULTI_GAIT:
                froude = command_speed.square() / (GRAVITY_M_S2 * LEG_LENGTH_M)
                initial_mode[froude > FROUDE_THRESHOLD] = int(GaitMode.RUN)
            initial_mode[static[ids]] = int(GaitMode.STAND)
            self.gait_mode[ids] = initial_mode

        root_speed = torch.linalg.vector_norm(_base(self._env)[0][ids, :2], dim=-1)
        self.previous_speed[ids] = root_speed
        return metrics

    def compute(self, dt: float):
        """更新速度命令、参考序列、步态相位和状态机。

        Args:
            dt: 本次策略控制周期 [s]，当前通常为 0.02 s（50 Hz）。

        Notes:
            相位每个策略周期更新；模式切换累计到 0.1 s 后才检查一次，即 10 Hz。
            这种分频不能简单替换成“每五步更新”，因为未来控制周期可能改变。
        """

        super().compute(dt)
        if self.reference_buffer is not None:
            # 命令项以策略控制频率调用。显式换算参考序列步数，避免未来修改
            # decimation 后，录制目标和奖励时间轴悄悄失步。
            steps = max(1, int(round(dt / float(getattr(self.cfg, "reference_target_dt_s", 0.02)))))
            self.reference_buffer.advance(steps)
        self._advance_phase(dt)
        self._gait_update_accumulator += dt
        if self._gait_update_accumulator + 1.0e-9 >= GAIT_UPDATE_PERIOD_S:
            elapsed = self._gait_update_accumulator
            self._gait_update_accumulator = 0.0
            self._advance_mode(elapsed)

    def set_curriculum_phase(self, phase: CurriculumPhase):
        """设置课程阶段，并清除该阶段尚未开放的步态标签。

        Args:
            phase: Walk-only、Transitions 或 Multi-gait 阶段。

        Notes:
            阶段回退时必须同步改写已有环境标签，否则奖励掩码仍会激活尚未开放的
            Run 等奖励。静止任务覆盖在所有阶段都保持 Stand 且相位为零。
        """

        self.curriculum_phase = CurriculumPhase(phase)
        static = self._static_task_mask()
        if self.curriculum_phase == CurriculumPhase.WALK_ONLY:
            self.gait_mode[:] = torch.where(
                static,
                torch.full_like(self.gait_mode, int(GaitMode.STAND)),
                torch.full_like(self.gait_mode, int(GaitMode.WALK)),
            )
        elif self.curriculum_phase == CurriculumPhase.TRANSITIONS:
            unavailable = (self.gait_mode == int(GaitMode.RUN)) | (self.gait_mode == int(GaitMode.RUN_TO_WALK))
            self.gait_mode[unavailable] = int(GaitMode.WALK)
        self.gait_mode[static] = int(GaitMode.STAND)
        self.gait_phase[static] = 0.0

    def set_task_speed(self, env_ids: Sequence[int] | torch.Tensor, speed_m_s: torch.Tensor | float) -> None:
        """把选中环境固定到数据集任务规定的前进速度。

        Args:
            env_ids: 要更新的环境下标，shape 为 ``[K]``。
            speed_m_s: 前进目标速度 [m/s]；可以是标量或每个环境一个值 ``[K]``。

        Raises:
            ValueError: 速度数量与环境数量不同，或包含 NaN/无穷大。

        Notes:
            覆盖会立即写入命令，并在以后每次命令重采样后再次写入。没有调用本方法
            的普通平地任务仍使用原课程范围。
        """

        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        speeds = torch.as_tensor(speed_m_s, device=self.device, dtype=torch.float32).reshape(-1)
        if speeds.numel() == 1:
            speeds = speeds.expand(ids.numel())
        if speeds.numel() != ids.numel():
            raise ValueError("speed_m_s must be scalar or have one value per environment ID")
        if not torch.isfinite(speeds).all():
            raise ValueError("task-specific speed targets must be finite")
        self.task_speed_m_s[ids] = speeds
        self.vel_command_b[ids, 0] = speeds
        self.vel_command_b[ids, 1:] = 0.0
        static = speeds.abs() < 0.1
        self.is_standing_env[ids] = static
        self.gait_mode[ids] = torch.where(
            static,
            torch.full_like(self.gait_mode[ids], int(GaitMode.STAND)),
            torch.full_like(self.gait_mode[ids], int(GaitMode.WALK)),
        )
        self.gait_phase[ids] = torch.where(static, torch.zeros_like(self.gait_phase[ids]), self.gait_phase[ids])

    def clear_task_speed(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """清除任务速度覆盖，恢复配置/课程控制的命令采样。

        Args:
            env_ids: 要恢复的环境下标，shape 为 ``[K]``。
        """

        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel():
            self.task_speed_m_s[ids] = float("nan")
            self.gait_mode[ids] = int(GaitMode.WALK)

    def _resample_command(self, env_ids: Sequence[int]):
        """先调用父类采样命令，再重新应用有效的模仿任务速度覆盖。

        Args:
            env_ids: 要重采样的环境下标或 slice。

        Notes:
            若不在父类采样后重写，数秒一次的随机命令会覆盖数据集指定速度，使目标
            运动与教师序列不再同步。
        """

        if isinstance(env_ids, slice):
            start, stop, step = env_ids.indices(self.num_envs)
            ids = torch.arange(start, stop, step, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        super()._resample_command(ids)
        speeds = self.task_speed_m_s[ids]
        valid = torch.isfinite(speeds)
        if valid.any():
            selected = ids[valid]
            self.vel_command_b[selected, 0] = speeds[valid]
            self.vel_command_b[selected, 1:] = 0.0
            selected_static = speeds[valid].abs() < 0.1
            self.is_standing_env[selected] = selected_static
            self.gait_mode[selected] = torch.where(
                selected_static,
                torch.full_like(self.gait_mode[selected], int(GaitMode.STAND)),
                torch.full_like(self.gait_mode[selected], int(GaitMode.WALK)),
            )
            self.gait_phase[selected] = torch.where(
                selected_static, torch.zeros_like(self.gait_phase[selected]), self.gait_phase[selected]
            )

    def _resample(self, env_ids: Sequence[int]):
        """把全量 slice 统一成下标张量后调用父类重采样流程。

        Args:
            env_ids: 环境下标序列或父类全量重置传入的 slice。
        """

        if isinstance(env_ids, slice):
            start, stop, step = env_ids.indices(self.num_envs)
            env_ids = torch.arange(start, stop, step, device=self.device, dtype=torch.long)
        super()._resample(env_ids)

    def _advance_phase(self, dt: float):
        """按当前步态的节奏推进每个环境的周期相位。

        Args:
            dt: 策略控制周期 [s]。

        Notes:
            cadence 单位为周期每秒 [Hz]，所以相位增量为 ``2*pi*cadence*dt``。
            Stand 的 cadence 为零，并额外把相位固定为零，避免静止奖励随相位变化。
        """

        cadence = torch.zeros(self.num_envs, device=self.device)
        cadence[self.gait_mode == int(GaitMode.WALK)] = 0.8
        cadence[self.gait_mode == int(GaitMode.WALK_TO_STAND)] = 0.6
        cadence[self.gait_mode == int(GaitMode.RUN)] = 1.6
        cadence[self.gait_mode == int(GaitMode.RUN_TO_WALK)] = 1.0
        self.gait_phase.add_(2.0 * math.pi * cadence * dt).remainder_(2.0 * math.pi)
        self.gait_phase[self.gait_mode == int(GaitMode.STAND)] = 0.0

    def _advance_mode(self, dt: float):
        """以 10 Hz 检查并执行五步态状态转换。

        Args:
            dt: 自上次模式更新以来累计的时间 [s]，通常约 0.1 s。

        状态机原因：
            - Walk-only 阶段只允许 Walk，静止数据任务除外；
            - 低速命令先进入 Walk-to-Stand，只有双脚支撑且机身低速连续保持
              ``WALK_TO_STAND_HOLD_S`` 才进入 Stand，避免经过零速度时误判停稳；
            - Multi-gait 阶段用 Froude 数判断 Run；Run 收到走路命令后先进入
              Run-to-Walk，保持 ``RUN_TO_WALK_HOLD_S`` 后才变为 Walk；
            - 数据集静止任务拥有最高优先级，始终覆盖成 Stand。
        """

        static = self._static_task_mask()
        if self.curriculum_phase == CurriculumPhase.WALK_ONLY:
            self.gait_mode[:] = torch.where(
                static,
                torch.full_like(self.gait_mode, int(GaitMode.STAND)),
                torch.full_like(self.gait_mode, int(GaitMode.WALK)),
            )
            self.gait_phase[static] = 0.0
            self.previous_speed[:] = torch.linalg.vector_norm(_base(self._env)[0][:, :2], dim=-1)
            return

        command_speed = torch.linalg.vector_norm(self.command[:, :2], dim=-1)
        root_speed = torch.linalg.vector_norm(_base(self._env)[0][:, :2], dim=-1)
        left_force, right_force = _foot_contacts(self._env)
        double_support = (left_force > CONTACT_FORCE_THRESHOLD_N) & (right_force > CONTACT_FORCE_THRESHOLD_N)
        low_command = (command_speed < 0.1) & ~static

        entering_stop = (
            low_command & (self.gait_mode != int(GaitMode.STAND)) & (self.gait_mode != int(GaitMode.WALK_TO_STAND))
        )
        self.gait_mode[entering_stop] = int(GaitMode.WALK_TO_STAND)
        self.transition_time[entering_stop] = 0.0
        self.stable_time[entering_stop] = 0.0

        stopping = low_command & (self.gait_mode == int(GaitMode.WALK_TO_STAND))
        stable_stop = stopping & double_support & (root_speed < 0.1)
        self.stable_time[stopping] = torch.where(
            stable_stop[stopping], self.stable_time[stopping] + dt, torch.zeros_like(self.stable_time[stopping])
        )
        stopped = stopping & (self.stable_time >= WALK_TO_STAND_HOLD_S)
        self.gait_mode[stopped] = int(GaitMode.STAND)
        self.gait_phase[stopped] = 0.0

        moving_command = (~low_command) & ~static
        if self.curriculum_phase == CurriculumPhase.TRANSITIONS:
            self.gait_mode[moving_command] = int(GaitMode.WALK)
        else:
            froude = command_speed.square() / (GRAVITY_M_S2 * LEG_LENGTH_M)
            run_command = moving_command & (froude > FROUDE_THRESHOLD)
            self.gait_mode[run_command] = int(GaitMode.RUN)

            walk_command = moving_command & ~run_command
            entering_r2w = walk_command & (self.gait_mode == int(GaitMode.RUN))
            self.gait_mode[entering_r2w] = int(GaitMode.RUN_TO_WALK)
            self.transition_time[entering_r2w] = 0.0

            transitioning = walk_command & (self.gait_mode == int(GaitMode.RUN_TO_WALK))
            self.transition_time[transitioning] += dt
            completed = transitioning & (self.transition_time >= RUN_TO_WALK_HOLD_S)
            self.gait_mode[completed] = int(GaitMode.WALK)

            regular_walk = walk_command & ~transitioning & ~entering_r2w
            self.gait_mode[regular_walk] = int(GaitMode.WALK)

        self.transition_time[self.gait_mode == int(GaitMode.WALK_TO_STAND)] += dt
        self.gait_mode[static] = int(GaitMode.STAND)
        self.gait_phase[static] = 0.0
        self.transition_time[static] = 0.0
        self.stable_time[static] = 0.0
        self.previous_speed[:] = root_speed

    @property
    def reference_sample(self) -> dict[str, torch.Tensor] | None:
        """返回当前逐环境传感器参考；未加载参考文件时返回 ``None``。

        Returns:
            参考通道字典。典型关节张量 shape 为 ``[num_envs, 8]``；是否存在速度、
            IMU 和接触通道由输入录制文件决定。
        """

        return None if self.reference_buffer is None else self.reference_buffer.sample()


# -----------------------------------------------------------------------------
# 传感器读取与 Actor/Critic 观测
# -----------------------------------------------------------------------------


def _sensor_stats(sensor, env_count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """汇总一个足底接触传感器在每个环境中的法向力统计量。

    Args:
        sensor: 必须提供 ``data.net_normal_forces_w`` 的 ContactSensor。
        env_count: 并行环境数量 ``N``。
        device: 输出张量所在的 Torch 设备。

    Returns:
        ``(总和, 均值, 标准差)``，每项单位为牛顿 [N]、shape 均为 ``[N]``。

    Raises:
        RuntimeError: 传感器没有法向力缓冲区。
        ValueError: 原始力不是 ``[N, num_sensors, 3]``。
    """
    try:
        force_value = sensor.data.net_normal_forces_w
    except AttributeError as exc:
        raise RuntimeError("foot ContactSensor must expose net_normal_forces_w") from exc
    if force_value is None:
        raise RuntimeError("foot ContactSensor returned no net_normal_forces_w buffer")
    force = _as_tensor(force_value)
    if force.ndim != 3 or force.shape[0] != env_count or force.shape[-1] != 3:
        raise ValueError(
            "foot ContactSensor net_normal_forces_w must have shape "
            f"({env_count}, num_sensors, 3), got {tuple(force.shape)}"
        )
    if force.device != device:
        force = force.to(device=device)
    magnitudes = force.reshape(env_count, -1, 3).norm(dim=-1)
    return magnitudes.sum(dim=-1), magnitudes.mean(dim=-1), magnitudes.std(dim=-1, unbiased=False)


def _foot_contacts(env) -> tuple[torch.Tensor, torch.Tensor]:
    """读取左右脚接触力幅值。

    Args:
        env: 包含左右足底 ContactSensor 的环境。

    Returns:
        ``(左脚力, 右脚力)``，单位为牛顿 [N]，每项 shape 为 ``[num_envs]``。
        接触通道关闭时返回全零，保持观测维度和 checkpoint 契约不变。
    """
    if not _optional_sensor_enabled(env, "contact"):
        zeros = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        return zeros, zeros.clone()
    left = _sensor_stats(env.scene["left_foot_contact"], env.num_envs, env.device)[0]
    right = _sensor_stats(env.scene["right_foot_contact"], env.num_envs, env.device)[0]
    return left, right


def _foot_pressure_features(env) -> torch.Tensor:
    """构造左右脚接触压力代理特征。

    Returns:
        按“左均值、左标准差、右均值、右标准差”排列的张量 [N]，shape 为
        ``[num_envs, 4]``。接触通道关闭时为全零。
    """
    if not _optional_sensor_enabled(env, "contact"):
        return torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
    left = _sensor_stats(env.scene["left_foot_contact"], env.num_envs, env.device)
    right = _sensor_stats(env.scene["right_foot_contact"], env.num_envs, env.device)
    return torch.stack((left[1], left[2], right[1], right[2]), dim=-1)


def _reference_sample(env) -> dict[str, torch.Tensor] | None:
    """从步态命令项取得当前录制参考样本。

    Returns:
        当前逐环境参考通道字典；命令项不支持参考数据或未启用时返回 ``None``。
    """

    try:
        sample = _gait_command(env).reference_sample
    except (AttributeError, TypeError):
        return None
    return sample


def _imu_gyro_features(env) -> torch.Tensor:
    """按规范传感器顺序读取六个连杆 IMU 的角速度。

    Returns:
        IMU 角速度 [rad/s]，shape 为 ``[num_envs, 6, 3]``。通道关闭时返回全零；
        旧场景缺少连杆 IMU 时重复根节点角速度作为兼容回退。

    Raises:
        ValueError: 已找到的 IMU 张量不是 ``[N, 3]`` 或 ``[N, 1, 3]``。

    Notes:
        根节点回退只为旧场景和轻量测试保留。正式模仿训练应提供全部六个 IMU，
        否则六个位置会收到相同信号，不能代表真实连杆运动。
    """

    root_gyro = _base(env)[1]
    if not _optional_sensor_enabled(env, "imu"):
        return torch.zeros(
            (root_gyro.shape[0], len(getattr(env, "imitation_imu_sensor_names", IMU_SENSOR_NAMES)), 3),
            device=root_gyro.device,
            dtype=root_gyro.dtype,
        )
    # 模仿任务使用采集器编号 imu_01..imu_06，旧平地任务仍保留解剖学别名；优先
    # 使用环境显式声明的顺序，场景没有创建连杆 IMU 时才回退到根节点角速度。
    sensor_names = tuple(getattr(env, "imitation_imu_sensor_names", IMU_SENSOR_NAMES))
    readings = []
    for sensor_name in sensor_names:
        try:
            value = _as_tensor(env.scene[sensor_name].data.ang_vel_b)
        except (AttributeError, KeyError, TypeError):
            return root_gyro[:, None, :].expand(-1, len(sensor_names), -1)
        if value.ndim == 3:
            value = value[:, 0, :]
        if value.ndim != 2 or value.shape[-1] != 3 or value.shape[0] != root_gyro.shape[0]:
            raise ValueError(f"IMU {sensor_name} angular velocity must have shape [N, 3] or [N, 1, 3]")
        readings.append(value)
    return torch.stack(readings, dim=1)


def sensor_reference_imitation_reward(
    env,
    joint_pos_weight: float = 1.0,
    joint_vel_weight: float = 0.25,
    gyro_weight: float = 0.5,
    contact_weight: float = 0.25,
    joint_pos_sigma: float = 0.35,
    joint_vel_sigma: float = 1.5,
    gyro_sigma: float = 1.0,
    contact_sigma: float = 0.35,
):
    """计算仿真状态与可选录制轨迹的一致性奖励。

    :class:`GaitVelocityCommand` 以策略频率推进参考。缺失或关闭的可选通道不参与
    计算，剩余权重重新归一化，因此只有关节角的数据也可以使用。各通道先按 sigma
    缩放平方误差，再通过指数核映射到 ``[0, 1]``。

    Args:
        env: 并行强化学习环境。
        joint_pos_weight: 关节角通道权重。
        joint_vel_weight: 关节角速度通道权重。
        gyro_weight: 六 IMU 角速度通道权重。
        contact_weight: 左右脚接触标签通道权重。
        joint_pos_sigma: 关节角误差尺度 [rad]。
        joint_vel_sigma: 关节角速度误差尺度 [rad/s]。
        gyro_sigma: IMU 角速度误差尺度 [rad/s]。
        contact_sigma: 二值接触误差尺度，无单位。

    Returns:
        每环境模仿相似度，shape 为 ``[num_envs]``。未加载参考时返回全零。

    Raises:
        ValueError: 权重为负、全部权重为零，或任一 sigma 不为正。

    Notes:
        这是训练信号，不应配合关节顺序、零位或传感器坐标系未经标定的录制数据。
    """

    weights = (joint_pos_weight, joint_vel_weight, gyro_weight, contact_weight)
    if any(weight < 0.0 for weight in weights) or not any(weight > 0.0 for weight in weights):
        raise ValueError("imitation weights must be non-negative and contain at least one positive value")
    sigmas = (joint_pos_sigma, joint_vel_sigma, gyro_sigma, contact_sigma)
    if any(sigma <= 0.0 for sigma in sigmas):
        raise ValueError("imitation sigmas must be positive")

    joint_pos = _canonical_joint_tensor(env, env.scene["robot"].data.joint_pos)
    zero = torch.zeros(joint_pos.shape[0], device=joint_pos.device, dtype=joint_pos.dtype)
    sample = _reference_sample(env)
    if sample is None:
        return zero

    scores = []
    active_weights = []
    reference_joint_pos = sample["joint_pos_rad"].to(dtype=joint_pos.dtype)
    scores.append(torch.exp(-torch.mean(torch.square((joint_pos - reference_joint_pos) / joint_pos_sigma), dim=-1)))
    active_weights.append(joint_pos_weight)

    if joint_vel_weight > 0.0 and "joint_vel_rad_s" in sample:
        joint_vel = _canonical_joint_tensor(env, env.scene["robot"].data.joint_vel)
        reference_joint_vel = sample["joint_vel_rad_s"].to(dtype=joint_vel.dtype)
        scores.append(torch.exp(-torch.mean(torch.square((joint_vel - reference_joint_vel) / joint_vel_sigma), dim=-1)))
        active_weights.append(joint_vel_weight)

    if gyro_weight > 0.0 and "gyro_rad_s" in sample and _optional_sensor_enabled(env, "imu"):
        gyro = _imu_gyro_features(env)
        reference_gyro = sample["gyro_rad_s"].to(dtype=gyro.dtype)
        scores.append(torch.exp(-torch.mean(torch.square((gyro - reference_gyro) / gyro_sigma), dim=(-1, -2))))
        active_weights.append(gyro_weight)

    if contact_weight > 0.0 and "contact_lr" in sample and _optional_sensor_enabled(env, "contact"):
        left, right = _foot_contacts(env)
        contact = torch.stack((left > CONTACT_FORCE_THRESHOLD_N, right > CONTACT_FORCE_THRESHOLD_N), dim=-1).to(
            joint_pos.dtype
        )
        reference_contact = sample["contact_lr"].to(dtype=contact.dtype)
        scores.append(torch.exp(-torch.mean(torch.square((contact - reference_contact) / contact_sigma), dim=-1)))
        active_weights.append(contact_weight)

    # 当前实现总会先加入关节位置通道，即使它的权重是 0，所以此列表通常不会为空。
    # 这个分支只是为未来可能“跳过全部通道”的改动留保护，它并不检查权重之和是否为 0。
    if not active_weights:
        return zero
    total_weight = sum(active_weights)
    score = sum(weight * value for weight, value in zip(active_weights, scores, strict=True)) / total_weight
    if "gait_id" in sample:
        gait = _gait_command(env)
        reference_gait = sample["gait_id"].to(device=joint_pos.device)
        # 负 gait 标签表示“适用于所有模式”，供没有同步步态分类器的录制使用。
        valid = (reference_gait < 0) | (reference_gait == gait.gait_mode)
        score = score * valid.to(dtype=score.dtype)
    return score


def exoskeleton_observation(env) -> torch.Tensor:
    """拼接 Actor 使用的 49 维步态条件观测。

    Args:
        env: 包含机器人、动作管理器、速度命令和足底传感器的环境。

    Returns:
        shape 为 ``[num_envs, 49]`` 的观测，依次为：投影重力 3、机身角速度 3、
        机身线速度 3、关节角/角速度/上一归一化动作各 8、左右接触力各 1、左右脚
        压力均值/标准差 4、速度命令 3、步态相位 sin/cos 2、五步态 one-hot 5。

    Raises:
        ValueError: 上一动作不是与八关节位置相同的 ``[num_envs, 8]`` shape。

    Notes:
        ContactSensor 只给聚合力，不是论文/Gazebo 的 60 点压力阵列，因此这里用
        聚合力的均值和标准差作为固定维度代理。
    """
    robot = env.scene["robot"]
    data = robot.data
    gravity = _as_tensor(data.projected_gravity_b)
    ang = _as_tensor(data.root_ang_vel_b)
    lin = _as_tensor(data.root_lin_vel_b)
    pos = _canonical_joint_tensor(env, data.joint_pos)
    vel = _canonical_joint_tensor(env, data.joint_vel)
    # 论文使用“上一策略动作”。prev_action 是交给动作管理器的归一化值，不是位置
    # 目标；若把它与当前关节角相加，会混合 rad 与无量纲数值，物理意义错误。
    action_manager = getattr(env, "action_manager", None)
    previous_action = (
        torch.zeros_like(pos) if action_manager is None else _as_tensor(getattr(action_manager, "prev_action"))
    )
    if previous_action.ndim != 2 or previous_action.shape != pos.shape:
        raise ValueError(
            "my_exoskeleton expects one 8-DOF previous action, "
            f"got {tuple(previous_action.shape)} for joint positions {tuple(pos.shape)}"
        )
    left, right = _foot_contacts(env)
    commands = env.command_manager.get_command("base_velocity")
    pressure = _foot_pressure_features(env)
    gait = _gait_command(env)
    phase = torch.stack((torch.sin(gait.gait_phase), torch.cos(gait.gait_phase)), dim=-1)
    gait_one_hot = torch.nn.functional.one_hot(gait.gait_mode, num_classes=GAIT_MODE_COUNT).to(pos.dtype)
    return torch.cat(
        (
            gravity,
            ang,
            lin,
            pos,
            vel,
            previous_action,
            left[:, None],
            right[:, None],
            pressure,
            commands,
            phase,
            gait_one_hot,
        ),
        dim=-1,
    )


def exoskeleton_critic_observation(env) -> torch.Tensor:
    """构造只在训练期提供给 Critic 的 62 维特权观测。

    Returns:
        shape 为 ``[num_envs, 62]`` 的张量：Actor 的 49 维观测，加左右接触标签 2、
        归一化接触力 2、根节点高度 1 和八关节角加速度 8。部署 Actor 不需要这些
        特权量。
    """

    policy = exoskeleton_observation(env)
    robot = env.scene["robot"]
    left, right = _foot_contacts(env)
    contacts = torch.stack((left > CONTACT_FORCE_THRESHOLD_N, right > CONTACT_FORCE_THRESHOLD_N), dim=-1).to(
        policy.dtype
    )
    forces = torch.stack((left, right), dim=-1) / 500.0
    root_height = _as_tensor(robot.data.root_pos_w)[:, 2:3]
    joint_acc = _canonical_joint_tensor(env, robot.data.joint_acc)
    return torch.cat((policy, contacts, forces, root_height, joint_acc), dim=-1)


def _base(env):
    """取得机身根节点状态的 Torch 张量。

    Returns:
        ``(机身坐标线速度 [m/s], 机身坐标角速度 [rad/s], 世界坐标四元数)``；
        shape 分别为 ``[N, 3]``、``[N, 3]``、``[N, 4]``。
    """
    data = env.scene["robot"].data
    values = (data.root_lin_vel_b, data.root_ang_vel_b, data.root_quat_w)
    return tuple(_as_tensor(value) for value in values)


def _gait_command(env) -> GaitVelocityCommand:
    """取得带步态状态机的 ``base_velocity`` 命令项。

    Returns:
        当前环境的 :class:`GaitVelocityCommand`。

    Raises:
        TypeError: 配置使用了不含 ``gait_mode`` 的普通速度命令项。
    """

    term = env.command_manager.get_term("base_velocity")
    if not hasattr(term, "gait_mode"):
        raise TypeError("base_velocity must use GaitVelocityCommand")
    return term


def gait_mask(env, gait: GaitMode | int) -> torch.Tensor:
    """生成只激活指定步态环境的浮点掩码。

    Args:
        env: 并行环境。
        gait: 目标步态枚举或整数标签。

    Returns:
        值为 0 或 1 的张量，shape 为 ``[num_envs]``。
    """

    return _shared_gait_mask(env, gait)


def _masked(value: torch.Tensor, env, gait: GaitMode | int) -> torch.Tensor:
    """把逐环境奖励限制到指定步态，其他步态置零。

    Args:
        value: 原始逐环境奖励，shape 为 ``[num_envs]``。
        env: 并行环境。
        gait: 应激活该奖励的步态。

    Returns:
        应用同 shape 步态掩码后的奖励。
    """

    return value * gait_mask(env, gait).to(value.dtype)


def _locomotion_mask(env) -> torch.Tensor:
    """为走、跑及两种过渡步态返回 1，为 Stand 返回 0。

    Returns:
        浮点掩码，shape 为 ``[num_envs]``。轻量测试替身没有命令管理器时返回全 1。

    Notes:
        共用生物力学奖励不应鼓励 Stand 时摆腿，因此使用本掩码排除静止模式。
    """

    try:
        command = _gait_command(env)
    except (AttributeError, TypeError):
        return torch.ones(env.scene["robot"].data.joint_vel.shape[0], device=env.scene["robot"].data.joint_vel.device)
    modes = command.gait_mode
    return (
        (modes == int(GaitMode.WALK))
        | (modes == int(GaitMode.WALK_TO_STAND))
        | (modes == int(GaitMode.RUN))
        | (modes == int(GaitMode.RUN_TO_WALK))
    ).to(dtype=torch.float32)


def _forward_motion_gate(env, speed_scale: float = 0.2) -> torch.Tensor:
    """按实际正向速度生成 ``[0, 1]`` 活动门，原地策略严格为零。

    Args:
        env: 并行环境。
        speed_scale: 门控达到 1 所需的正向速度 [m/s]。

    Returns:
        shape 为 ``[num_envs]`` 的门控值。
    """

    if speed_scale <= 0.0:
        raise ValueError("speed_scale must be positive")
    forward_speed = _base(env)[0][:, 0]
    return torch.clamp(forward_speed / speed_scale, min=0.0, max=1.0)


def tracking_reward(env):
    """奖励平面线速度和偏航角速度同时跟踪命令。

    Returns:
        三项平方误差的指数相似度，shape 为 ``[num_envs]``，范围 ``(0, 1]``。
    """
    lin, ang, _ = _base(env)
    cmd = env.command_manager.get_command("base_velocity")
    error = torch.cat((lin[:, :2] - cmd[:, :2], ang[:, 2:3] - cmd[:, 2:3]), dim=-1)
    return torch.exp(-torch.sum(error.square(), dim=-1) / 0.25)


def tracking_linear_velocity_reward(
    env,
    std: float = 0.5,
    activity_gate_speed: float | None = None,
    activity_gate_command_max: float | None = None,
):
    """奖励平面速度命令跟踪，并可要求低速命令下确实向前运动。

    Args:
        env: 并行环境。
        std: 平面速度误差的指数核尺度 [m/s]。
        activity_gate_speed: 实际正向速度门控达到 1 的尺度 [m/s]。
        activity_gate_command_max: 只对不超过该幅值的低速命令启用活动门 [m/s]。

    Returns:
        每环境跟踪相似度，shape 为 ``[num_envs]``。

    Notes:
        低速门只服务固定前进课程，用于消除站着不动仍获得
        ``exp(-0.2**2/std**2)`` 的捷径，而不让高速命令分布变得过尖。
    """

    lin, _, _ = _base(env)
    command = env.command_manager.get_command("base_velocity")
    error = torch.sum(torch.square(lin[:, :2] - command[:, :2]), dim=-1)
    score = torch.exp(-error / (std * std))
    if activity_gate_speed is not None or activity_gate_command_max is not None:
        if activity_gate_speed is None or activity_gate_command_max is None:
            raise ValueError("activity gate requires both speed and command thresholds")
        if activity_gate_speed <= 0.0 or activity_gate_command_max <= 0.0:
            raise ValueError("activity gate thresholds must be positive")
        low_forward_command = (command[:, 0] > 0.0) & (
            torch.linalg.vector_norm(command[:, :2], dim=-1) <= activity_gate_command_max
        )
        activity = torch.clamp(lin[:, 0] / activity_gate_speed, min=0.0, max=1.0)
        score = torch.where(low_forward_command, score * activity, score)
    return score


def tracking_yaw_velocity_reward(env, std: float = 0.5):
    """用指数核奖励偏航角速度命令跟踪。

    Args:
        env: 并行环境。
        std: 偏航角速度误差尺度 [rad/s]。

    Returns:
        每环境相似度，shape 为 ``[num_envs]``。
    """

    _, ang, _ = _base(env)
    command = env.command_manager.get_command("base_velocity")
    return torch.exp(-torch.square(ang[:, 2] - command[:, 2]) / (std * std))


def fixed_forward_walk_reward(
    env,
    target_speed: float,
    speed_tolerance: float,
    lateral_speed_std: float,
    yaw_rate_std: float,
    command_speed_tolerance: float | None = None,
    gait: GaitMode | int | None = None,
):
    """奖励沿机身坐标正 x 方向以固定目标速度直行。

    速度窗在静止、倒退以及达到目标两倍速度时均为零，防止姿态和接触奖励让原地
    站立策略与真正行走策略获得接近总分。

    Args:
        env: 管理器式强化学习环境。
        target_speed: 机身正向目标速度 [m/s]，必须为正。
        speed_tolerance: 二次正向速度窗口的半宽 [m/s]。
        lateral_speed_std: 机身横向速度的高斯衰减尺度 [m/s]。
        yaw_rate_std: 机身偏航角速度的高斯衰减尺度 [rad/s]。
        command_speed_tolerance: 可选命令速度门控尺度 [m/s]；后续课程命令偏离固定
            目标时逐渐关闭本奖励。
        gait: 可选步态标签；传入时只给该步态计分。

    Returns:
        每个环境 ``[0, 1]`` 的奖励，shape 为 ``[num_envs]``。
    """
    lin_vel_b, ang_vel_b, _ = _base(env)
    forward_speed = lin_vel_b[:, 0]
    forward_score = torch.clamp(1.0 - torch.square((forward_speed - target_speed) / speed_tolerance), min=0.0, max=1.0)
    straight_score = torch.exp(
        -torch.square(lin_vel_b[:, 1] / lateral_speed_std) - torch.square(ang_vel_b[:, 2] / yaw_rate_std)
    )
    score = forward_score * straight_score
    if command_speed_tolerance is not None and hasattr(env, "command_manager"):
        command = env.command_manager.get_command("base_velocity")
        if command_speed_tolerance <= 0.0:
            raise ValueError("command_speed_tolerance must be positive")
        command_gate = torch.exp(-torch.square((command[:, 0] - target_speed) / command_speed_tolerance))
        score = score * command_gate
    if gait is not None and hasattr(env, "command_manager"):
        score = _masked(score, env, gait)
    return score


# -----------------------------------------------------------------------------
# 共享运动学、接触与生物力学奖励
# -----------------------------------------------------------------------------


def lin_vel_z_reward(env):
    """返回机身竖直速度绝对值，供配置中的负权重形成惩罚。

    Returns:
        竖直速度幅值 [m/s]，shape 为 ``[num_envs]``。
    """

    return torch.abs(_base(env)[0][:, 2])


def ang_vel_xy_reward(env):
    """返回滚转和俯仰角速度绝对值之和，供负权重惩罚晃动。

    Returns:
        角速度幅值 [rad/s]，shape 为 ``[num_envs]``。
    """

    return torch.abs(_base(env)[1][:, :2]).sum(dim=-1)


def orientation_reward(env):
    """根据机身滚转角和俯仰角计算直立分数。

    Returns:
        ``1 - |roll| - |pitch|``，shape 为 ``[num_envs]``。倾斜很大时可为负值。
    """

    roll, pitch, _ = euler_xyz_from_quat(_base(env)[2])
    return 1.0 - torch.abs(roll) - torch.abs(pitch)


def torque_reward(env):
    """返回所有执行关节力矩平方和，由配置负权重转换为能耗惩罚。

    Returns:
        力矩平方和 [(N*m)^2]，shape 为 ``[num_envs]``。函数本身返回正成本，
        ``RewardTermCfg`` 当前使用 ``-1e-6`` 权重。
    """

    effort = _as_tensor(env.scene["robot"].data.applied_torque)
    return effort.square().sum(dim=-1)


def action_rate_reward(env):
    """返回相邻策略动作变化的平方和，抑制指令抖动。

    Returns:
        无量纲归一化动作变化成本，shape 为 ``[num_envs]``；配置用负权重形成惩罚。
    """

    action = env.action_manager.action
    previous = env.action_manager.prev_action
    return (action - previous).square().sum(dim=-1)


def contact_reward(env):
    """奖励至少一只脚有效接触地面，但不奖励撞击力大小。

    Returns:
        二值奖励，shape 为 ``[num_envs]``。任一脚法向力大于阈值时为 1。

    Notes:
        传感器原始单位是牛顿。若直接返回原始力，重踩地面的奖励会比速度跟踪大
        数个数量级，因此本项刻意限制为 ``[0, 1]``。
    """
    left, right = _foot_contacts(env)
    return ((left > CONTACT_FORCE_THRESHOLD_N) | (right > CONTACT_FORCE_THRESHOLD_N)).float()


def foot_contact_alternation_reward(env):
    """奖励左右脚恰好一只着地，鼓励交替支撑。

    Returns:
        左右接触状态异或（XOR）的二值奖励，shape 为 ``[num_envs]``。
    """

    left, right = _foot_contacts(env)
    left_on, right_on = left > CONTACT_FORCE_THRESHOLD_N, right > CONTACT_FORCE_THRESHOLD_N
    return (left_on ^ right_on).float()


def feet_forward_alignment_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    sigma: float,
    min_forward_speed: float,
    max_lateral_speed: float,
    max_yaw_rate: float,
):
    """直行时奖励双脚朝向命令规定的前进方向。

    速度命令位于机身坐标系，正向直行对应世界系中的根节点偏航方向。这里只比较
    yaw（偏航）；左右脚连杆的 roll 坐标轴互为镜像，即使前向轴平行也不能直接比较
    完整四元数。

    Args:
        env: 并行环境。
        asset_cfg: 指定机器人和有序双脚刚体的实体配置。
        command_name: 速度命令项名称。
        sigma: 双脚偏航误差的指数衰减尺度。
        min_forward_speed: 启用奖励所需的最小前向命令 [m/s]。
        max_lateral_speed: 仍视为直行的最大横向命令绝对值 [m/s]。
        max_yaw_rate: 仍视为直行的最大偏航命令绝对值 [rad/s]。

    Returns:
        双脚平均朝向相似度；非直行命令时为零，shape 为 ``[num_envs]``。
    """
    robot = env.scene[asset_cfg.name]
    foot_quat = _as_tensor(robot.data.body_link_quat_w)
    foot_quat = foot_quat[:, asset_cfg.body_ids]

    root_quat = _as_tensor(robot.data.root_quat_w)

    foot_yaw = math_utils.yaw_quat(foot_quat)
    desired_yaw = math_utils.yaw_quat(root_quat).unsqueeze(1).expand_as(foot_yaw)
    relative_yaw = math_utils.quat_mul(math_utils.quat_conjugate(desired_yaw), foot_yaw)
    yaw_error = relative_yaw[..., :3].square().sum(dim=-1)
    alignment = torch.exp(-yaw_error / (2.0 * sigma * sigma)).mean(dim=1)

    command = env.command_manager.get_command(command_name)
    is_straight_forward = (
        (command[:, 0] > min_forward_speed)
        & (command[:, 1].abs() <= max_lateral_speed)
        & (command[:, 2].abs() <= max_yaw_rate)
    )
    return alignment * is_straight_forward.float()


def _centroidal_angular_momentum_z(data) -> tuple[torch.Tensor, torch.Tensor]:
    """计算系统相对总质心的世界 z 轴角动量。

    Args:
        data: 机器人 articulation 数据，必须包含刚体质量、质心位置/速度、惯量和
            连杆世界姿态。

    Returns:
        ``(总 z 角动量, 每刚体三维角动量)``，单位均为 ``kg*m^2/s``，shape 分别为
        ``[num_envs]`` 和 ``[num_envs, num_bodies, 3]``。逐刚体结果用于替代论文中
        不适用于无手臂外骨骼的手臂项。

    Raises:
        ValueError: 仿真后端返回的刚体状态 shape 不完整或不一致。
    """

    mass = _as_tensor(data.body_mass)
    com_pos = _as_tensor(data.body_com_pos_w)
    com_vel = _as_tensor(data.body_com_vel_w)
    inertia = _as_tensor(data.body_inertia)
    body_link_quat_w = _as_tensor(data.body_link_quat_w)
    if mass.ndim != 2 or com_pos.ndim != 3 or com_vel.ndim != 3 or inertia.ndim != 3:
        raise ValueError("unexpected articulation body-state shapes for centroidal momentum")
    if (
        com_vel.shape[-1] < 6
        or inertia.shape[-1] != 9
        or body_link_quat_w.shape[-1] != 4
        or mass.shape[:2] != com_pos.shape[:2]
        or body_link_quat_w.shape[:2] != mass.shape[:2]
    ):
        raise ValueError("incomplete articulation body state for centroidal momentum")

    mass = mass.clamp_min(1.0e-8)
    total_mass = mass.sum(dim=1, keepdim=True)
    centroid = (mass.unsqueeze(-1) * com_pos).sum(dim=1, keepdim=True) / total_mass.unsqueeze(-1)
    centroid_vel = (mass.unsqueeze(-1) * com_vel[..., :3]).sum(dim=1, keepdim=True) / total_mass.unsqueeze(-1)
    relative_pos = com_pos - centroid
    relative_vel = com_vel[..., :3] - centroid_vel
    orbital = torch.cross(relative_pos, mass.unsqueeze(-1) * relative_vel, dim=-1)
    # PhysX 把质心惯量表示为刚体 prim 坐标系中的九个列主序值；先转成行主序 3x3
    # 矩阵并旋转到世界系，再与世界角速度相乘。
    inertia_link = inertia.reshape(inertia.shape[0], inertia.shape[1], 3, 3).transpose(-1, -2)
    link_rotation_w = math_utils.matrix_from_quat(body_link_quat_w)
    inertia_world = link_rotation_w @ inertia_link @ link_rotation_w.transpose(-1, -2)
    spin = torch.einsum("nbij,nbj->nbi", inertia_world, com_vel[..., 3:6])
    body_momentum = orbital + spin
    total = body_momentum.sum(dim=1)[:, 2]
    return total, body_momentum


def _side_momentum_z(body_momentum: torch.Tensor, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """把有序刚体角动量均分为右腿和左腿两组。

    Args:
        body_momentum: 每刚体三维角动量 [kg*m^2/s]，shape 为 ``[N, B, 3]``。
        asset_cfg: ``body_ids`` 必须先列右腿刚体、再列等量左腿刚体。

    Returns:
        ``(左腿 z 角动量, 右腿 z 角动量)``，每项 shape 为 ``[N]``。

    Raises:
        ValueError: 选中刚体不足两个或数量不是偶数。
    """

    ids = asset_cfg.body_ids
    selected = body_momentum if isinstance(ids, slice) else body_momentum[:, ids]
    if selected.shape[1] < 2 or selected.shape[1] % 2:
        raise ValueError("side momentum requires an even, ordered body selection")
    midpoint = selected.shape[1] // 2
    right = selected[:, :midpoint, 2].sum(dim=-1)
    left = selected[:, midpoint:, 2].sum(dim=-1)
    return left, right


def leg_momentum_balance_reward(
    env,
    asset_cfg: SceneEntityCfg,
    momentum_scale: float = 10.0,
):
    """实现适配无手臂外骨骼后的角动量平衡项。

    论文公式包含总角动量成本和镜像左右手臂残差。本模型没有手臂，因此透明地改用
    有序左右腿刚体角动量，世界系残差为 ``-L_left,z - L_right,z``。总角动量仍由
    Isaac Lab 中真实质量、质心位置/速度和世界系惯量计算。

    Args:
        env: 并行环境。
        asset_cfg: 机器人及“右腿刚体在前、左腿刚体在后”的有序选择。
        momentum_scale: 角动量归一化尺度 [kg*m^2/s]。

    Returns:
        非正成本，shape 为 ``[num_envs]``；Stand 环境经活动掩码后为零。
    """

    if momentum_scale <= 0.0:
        raise ValueError("momentum_scale must be positive")
    data = env.scene[asset_cfg.name].data
    total_z, body_momentum = _centroidal_angular_momentum_z(data)
    left_z, right_z = _side_momentum_z(body_momentum, asset_cfg)
    # 论文在镜像局部手臂坐标中写侧向项。矢状面对称会翻转左侧 yaw 符号，因此世界
    # 系等价项为 (-L_left,z - L_right,z)^2，使反向转动的双腿相互抵消。
    mirrored_side_residual = -left_z - right_z
    residual = torch.square(total_z / momentum_scale) + 0.4 * torch.square(mirrored_side_residual / momentum_scale)
    score = -residual
    return score * _locomotion_mask(env).to(score.dtype)


def leg_mechanical_power_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    power_scale: float = 50.0,
):
    """计算归一化下肢机械功率成本。

    八自由度机构没有论文中的手臂，故以执行腿关节的 ``|tau*qdot|`` [W] 作为明确的
    下肢能耗代理。只在运动步态激活，避免把站立支撑力矩误认为低效摆动。

    Args:
        env: 并行环境。
        asset_cfg: 机器人及需要统计的执行关节。
        power_scale: 平均功率归一化尺度 [W]。

    Returns:
        截断到 ``[0, 10]`` 的正成本，shape 为 ``[num_envs]``；配置应使用负权重。
    """

    if power_scale <= 0.0:
        raise ValueError("power_scale must be positive")
    data = env.scene[asset_cfg.name].data
    joint_vel = _as_tensor(data.joint_vel)[:, asset_cfg.joint_ids]
    effort = getattr(data, "applied_torque", None)
    if effort is None:
        effort = getattr(data, "computed_torque", None)
    if effort is None:
        effort = torch.zeros_like(_as_tensor(data.joint_vel))
    effort = _as_tensor(effort)[:, asset_cfg.joint_ids]
    power = torch.nan_to_num(torch.abs(effort * joint_vel), nan=0.0, posinf=power_scale, neginf=power_scale)
    normalized = torch.clamp(power.mean(dim=-1) / power_scale, min=0.0, max=10.0)
    return normalized * _locomotion_mask(env).to(normalized.dtype)


def leg_phase_symmetry_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sigma: float = 0.5,
    activity_scale: float = 0.5,
    motion_scale: float = 0.2,
):
    """奖励左右腿反相运动，作为无手臂模型的对称性项。

    Args:
        env: 并行环境。
        asset_cfg: 按右腿四关节、左腿四关节排列的实体配置。
        sigma: 左右对应关节速度和的误差尺度 [rad/s]。
        activity_scale: 关节活动量的饱和尺度 [rad/s]。
        motion_scale: 正向运动门达到 1 的速度尺度 [m/s]。

    Returns:
        只在实际前进且处于运动步态时非零的奖励，shape 为 ``[num_envs]``。
    """

    if sigma <= 0.0 or activity_scale <= 0.0:
        raise ValueError("sigma and activity_scale must be positive")
    joint_vel = _as_tensor(env.scene[asset_cfg.name].data.joint_vel)[:, asset_cfg.joint_ids]
    if joint_vel.shape[-1] != 8:
        raise ValueError("leg_phase_symmetry_reward requires four ordered joints per leg")
    pair_error = torch.mean(torch.square(joint_vel[:, :4] + joint_vel[:, 4:8]), dim=-1)
    activity = torch.tanh(torch.mean(torch.abs(joint_vel), dim=-1) / activity_scale)
    score = activity * torch.exp(-pair_error / (sigma * sigma))
    return score * _forward_motion_gate(env, motion_scale) * _locomotion_mask(env).to(score.dtype)


def leg_amplitude_match_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sigma: float = 0.5,
    activity_scale: float = 0.5,
    motion_scale: float = 0.2,
):
    """奖励左右腿具有接近的摆动速度幅值。

    Args:
        env: 并行环境。
        asset_cfg: 按右腿四关节、左腿四关节排列的实体配置。
        sigma: 左右幅值差的指数核尺度 [rad/s]。
        activity_scale: 总活动量的饱和尺度 [rad/s]。
        motion_scale: 正向运动门达到 1 的速度尺度 [m/s]。

    Returns:
        只在实际前进且处于运动步态时非零的奖励，shape 为 ``[num_envs]``。
    """

    if sigma <= 0.0 or activity_scale <= 0.0:
        raise ValueError("sigma and activity_scale must be positive")
    joint_vel = _as_tensor(env.scene[asset_cfg.name].data.joint_vel)[:, asset_cfg.joint_ids]
    if joint_vel.shape[-1] != 8:
        raise ValueError("leg_amplitude_match_reward requires four ordered joints per leg")
    right_amp = torch.linalg.vector_norm(joint_vel[:, :4], dim=-1) / 2.0
    left_amp = torch.linalg.vector_norm(joint_vel[:, 4:8], dim=-1) / 2.0
    activity = torch.tanh((right_amp + left_amp) / (2.0 * activity_scale))
    score = activity * torch.exp(-torch.square(right_amp - left_amp) / (sigma * sigma))
    return score * _forward_motion_gate(env, motion_scale) * _locomotion_mask(env).to(score.dtype)


def phase_contact_reward(
    env,
    gait: GaitMode | int,
    allow_double_support: bool = False,
    flight_window: float = 0.0,
    double_support_ramp_s: float = 0.0,
):
    """奖励当前步态相位所要求的左右脚接触模式。

    Walk 要求精确单脚支撑；过渡步态可允许双脚支撑；Run 可在支撑交换附近保留
    短暂腾空窗口，从而表达论文中的支撑/腾空交替。``flight_window`` 是
    ``abs(sin(gait_phase))`` 的阈值，范围必须是 ``[0, 1)``。

    Args:
        env: 并行环境。
        gait: 本奖励对应的步态标签。
        allow_double_support: 是否把双脚支撑视为可接受模式。
        flight_window: Run 中允许双脚离地的相位正弦阈值。
        double_support_ramp_s: 双支撑得分从零增加到一所需的过渡时间 [s]；零表示
            立即允许。

    Returns:
        接触模式匹配分数，shape 为 ``[num_envs]``，并只在指定步态激活。
    """

    if not 0.0 <= flight_window < 1.0:
        raise ValueError("flight_window must be in [0, 1)")
    if double_support_ramp_s < 0.0:
        raise ValueError("double_support_ramp_s must be non-negative")

    left, right = _foot_contacts(env)
    observed = torch.stack((left > CONTACT_FORCE_THRESHOLD_N, right > CONTACT_FORCE_THRESHOLD_N), dim=-1)
    phase = _gait_command(env).gait_phase
    phase_sine = torch.sin(phase)
    expected_left = phase_sine >= 0.0
    expected = torch.stack((expected_left, ~expected_left), dim=-1)
    single_support_match = torch.all(observed == expected, dim=-1).float()
    # Walk-to-Stand 可能需要短暂双脚落地重整接触时序；Walk 不启用此选项，保持
    # 精确单脚支撑目标。
    score = single_support_match
    if allow_double_support:
        double_support = torch.all(observed, dim=-1).to(score.dtype)
        if double_support_ramp_s > 0.0:
            transition_time = _gait_command(env).transition_time
            blend = torch.clamp(transition_time / double_support_ramp_s, min=0.0, max=1.0).to(score.dtype)
            score = score * (1.0 - blend) + double_support * blend
        else:
            score = torch.maximum(score, double_support)
    if flight_window > 0.0:
        # 只有 Run 在支撑交换附近允许有界腾空；其他步态即使误传参数也必须报错，
        # 不能悄悄改变其接触语义。
        if int(gait) != int(GaitMode.RUN):
            raise ValueError("flight_window is only valid for the RUN gait")
        flight_phase = phase_sine.abs() < flight_window
        flight_match = (~torch.any(observed, dim=-1)).to(score.dtype)
        score = torch.where(flight_phase, flight_match, score)
    # 若对左右脚匹配取平均，双脚支撑在单支撑目标下仍恒得 0.5，形成站住不走的
    # 捷径；使用完整模式精确匹配可避免这一局部最优。
    return _masked(score, env, gait)


def swing_height_penalty(
    env,
    gait: GaitMode | int,
    asset_cfg: SceneEntityCfg,
    target_height: float,
):
    """计算指定步态中摆动脚高度误差成本。

    Args:
        env: 并行环境。
        gait: 本项对应的步态。
        asset_cfg: 按左脚、右脚选择足部刚体的配置。
        target_height: 摆动脚世界 z 方向目标高度 [m]。

    Returns:
        仅对离地脚平均的高度平方误差 [m^2]，shape 为 ``[num_envs]``，其他步态为零。
    """

    foot_pos = _as_tensor(env.scene[asset_cfg.name].data.body_link_pos_w)[:, asset_cfg.body_ids]
    left, right = _foot_contacts(env)
    airborne = torch.stack((left <= CONTACT_FORCE_THRESHOLD_N, right <= CONTACT_FORCE_THRESHOLD_N), dim=-1).float()
    error = torch.square(foot_pos[:, :, 2] - target_height) * airborne
    normalizer = airborne.sum(dim=-1).clamp(min=1.0)
    return _masked(error.sum(dim=-1) / normalizer, env, gait)


def straight_knee_reward(env, gait: GaitMode | int, asset_cfg: SceneEntityCfg, sigma: float = 0.35):
    """奖励指定步态中支撑腿膝关节接近伸直。

    Args:
        env: 并行环境。
        gait: 本项对应的步态。
        asset_cfg: 必须按右膝、左膝选择两个关节。
        sigma: 解剖学屈曲角误差尺度 [rad]。

    Returns:
        支撑腿伸直相似度，shape 为 ``[num_envs]``；无支撑脚或其他步态时为零。
    """

    joint_pos = _as_tensor(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids]
    if joint_pos.shape[-1] != 2:
        raise ValueError("straight_knee_reward requires ordered right and left knee joints")
    left_force, right_force = _foot_contacts(env)
    stance = torch.stack((right_force > CONTACT_FORCE_THRESHOLD_N, left_force > CONTACT_FORCE_THRESHOLD_N), dim=-1).to(
        joint_pos.dtype
    )
    signs = torch.as_tensor(
        [KNEE_ARTICULATION_SIGNS[name] for name in KNEE_JOINT_NAMES],
        device=joint_pos.device,
        dtype=joint_pos.dtype,
    )
    flexion = joint_pos * signs
    normalizer = stance.sum(dim=-1).clamp(min=1.0)
    error = (torch.square(flexion) * stance).sum(dim=-1) / normalizer
    score = torch.exp(-error / (sigma * sigma)) * (stance.sum(dim=-1) > 0.0).to(joint_pos.dtype)
    return _masked(score, env, gait)


# -----------------------------------------------------------------------------
# 膝关节安全与分步态专属奖励
# -----------------------------------------------------------------------------


def _knee_joint_names(asset_cfg: SceneEntityCfg) -> tuple[str, ...]:
    """取得管理器项显式声明的“右膝、左膝”关节顺序。

    Isaac Lab 在创建场景时解析 ``joint_ids``，但这里仍保留名称，以便仿真后端内部
    顺序改变时，安全项仍能把左右侧不同边界应用到正确膝关节。

    Args:
        asset_cfg: 包含 ``joint_names`` 的场景实体配置。

    Returns:
        固定为 ``KNEE_JOINT_NAMES`` 的二元素元组。

    Raises:
        ValueError: 名称缺失、数量错误或顺序不是右膝后左膝。
    """

    names = getattr(asset_cfg, "joint_names", None)
    if names is None:
        names = KNEE_JOINT_NAMES
    elif isinstance(names, str):
        names = (names,)
    else:
        names = tuple(names)
    if tuple(names) != KNEE_JOINT_NAMES:
        raise ValueError(
            f"knee terms require joint_names=['Rthigh_to_Rleg', 'Lthigh_to_Lleg'] in that order, got {tuple(names)!r}"
        )
    return tuple(names)


def _knee_bound_tensors(
    asset_cfg: SceneEntityCfg,
    device: torch.device,
    dtype: torch.dtype,
    bounds_by_joint: Mapping[str, tuple[float, float]] | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """解析左右膝在原始 articulation 坐标中的独立边界。

    Args:
        asset_cfg: 必须按右膝、左膝声明关节名的实体配置。
        device: 输出边界张量所在设备。
        dtype: 输出浮点类型。
        bounds_by_joint: 可选“关节名 -> (下界, 上界)”映射，单位 [rad]。
        lower: 兼容旧配置的统一下界 [rad]，必须与 ``upper`` 同时传入。
        upper: 兼容旧配置的统一上界 [rad]，必须与 ``lower`` 同时传入。

    Returns:
        ``(下界, 上界)``，每项 shape 为 ``[2]``，顺序为右膝、左膝。

    Raises:
        ValueError: 只传一个统一边界、缺少某侧边界，或下界不小于上界。

    Notes:
        左右膝原始轴方向互为镜像，正式配置应优先使用逐关节边界，而不是统一范围。
    """

    names = _knee_joint_names(asset_cfg)
    if bounds_by_joint is None:
        if (lower is None) != (upper is None):
            raise ValueError("lower and upper must be provided together")
        if lower is not None and upper is not None:
            bounds_by_joint = {name: (float(lower), float(upper)) for name in names}
        else:
            bounds_by_joint = KNEE_ARTICULATION_LIMITS_RAD
    try:
        bounds = [bounds_by_joint[name] for name in names]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing side-specific knee bound for {names!r}") from exc
    if any(len(pair) != 2 or float(pair[0]) >= float(pair[1]) for pair in bounds):
        raise ValueError(f"knee bounds must be increasing (lower, upper), got {bounds!r}")
    lower_tensor = torch.as_tensor([float(pair[0]) for pair in bounds], device=device, dtype=dtype)
    upper_tensor = torch.as_tensor([float(pair[1]) for pair in bounds], device=device, dtype=dtype)
    return lower_tensor, upper_tensor


def knee_limit_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    bounds_by_joint: Mapping[str, tuple[float, float]] | None = None,
    lower: float | None = None,
    upper: float | None = None,
):
    """计算膝角超出人体范围的正平方成本。

    Args:
        env: 并行环境。
        asset_cfg: 按右膝、左膝选择关节的实体配置。
        bounds_by_joint: 可选逐关节原始角度边界 [rad]。
        lower: 兼容旧配置的统一下界 [rad]。
        upper: 兼容旧配置的统一上界 [rad]。

    Returns:
        越界距离平方和 [rad^2]，shape 为 ``[num_envs]``；范围内为零。奖励管理器
        使用负权重把它转换为惩罚。

    Notes:
        边界采用原始 articulation 符号。左右膝轴互为镜像，单一全局区间可能错误
        地允许其中一侧反向弯曲。
    """

    joint_pos = _as_tensor(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids]
    lower_tensor, upper_tensor = _knee_bound_tensors(
        asset_cfg, joint_pos.device, joint_pos.dtype, bounds_by_joint, lower, upper
    )
    below = torch.relu(lower_tensor.unsqueeze(0) - joint_pos)
    above = torch.relu(joint_pos - upper_tensor.unsqueeze(0))
    return torch.sum(torch.square(below) + torch.square(above), dim=-1)


def knee_out_of_human_range(
    env,
    asset_cfg: SceneEntityCfg,
    bounds_by_joint: Mapping[str, tuple[float, float]] | None = None,
    lower: float | None = None,
    upper: float | None = None,
):
    """标记任一膝关节离开对应人体边界的环境。

    Returns:
        布尔终止掩码，shape 为 ``[num_envs]``；任一侧越界即为 ``True``。
    """

    joint_pos = _as_tensor(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids]
    lower_tensor, upper_tensor = _knee_bound_tensors(
        asset_cfg, joint_pos.device, joint_pos.dtype, bounds_by_joint, lower, upper
    )
    return torch.any((joint_pos < lower_tensor.unsqueeze(0)) | (joint_pos > upper_tensor.unsqueeze(0)), dim=-1)


def knee_velocity_limit_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    max_velocity: float = 3.14,
):
    """计算膝关节角速度超过执行器安全阈值的平方成本。

    Args:
        env: 并行环境。
        asset_cfg: 需要检查的膝关节选择。
        max_velocity: 允许的角速度绝对值上限 [rad/s]，必须为正。

    Returns:
        超限部分平方和 [(rad/s)^2]，shape 为 ``[num_envs]``。
    """

    if max_velocity <= 0.0:
        raise ValueError("max_velocity must be positive")
    joint_vel = _as_tensor(env.scene[asset_cfg.name].data.joint_vel)[:, asset_cfg.joint_ids]
    excess = torch.relu(torch.abs(joint_vel) - float(max_velocity))
    return torch.sum(torch.square(excess), dim=-1)


def human_knee_gait_reward(
    env,
    gait: GaitMode | int,
    asset_cfg: SceneEntityCfg,
    stance_target: float = 0.0,
    swing_target: float = 0.8,
    stance_sigma: float = 0.30,
    swing_sigma: float = 0.45,
):
    """奖励支撑腿生理伸展和摆动腿生理屈曲。

    Args:
        env: 并行环境。
        gait: 本奖励对应的运动步态。
        asset_cfg: 必须按右膝、左膝包含两个关节。
        stance_target: 支撑期目标解剖学屈曲角 [rad]。
        swing_target: 摆动期目标解剖学屈曲角 [rad]。
        stance_sigma: 支撑期误差尺度 [rad]。
        swing_sigma: 摆动期误差尺度 [rad]。

    Returns:
        左右膝平均相似度，shape 为 ``[num_envs]``，只在指定 gait 激活。

    Notes:
        接触标签与原始关节角符号分开处理，再用显式左右符号转成解剖学屈曲角。
        这可防止反向弯膝也拿到正分，同时允许摆动期正常屈膝。
    """

    names = _knee_joint_names(asset_cfg)
    joint_pos = _as_tensor(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids]
    if joint_pos.shape[-1] != 2:
        raise ValueError("human_knee_gait_reward requires ordered right and left knee joints")
    # 用显式左右符号把原始 articulation 角转换为解剖学屈曲角，不从张量位置猜方向。
    signs = torch.as_tensor(
        [KNEE_ARTICULATION_SIGNS[name] for name in names],
        device=joint_pos.device,
        dtype=joint_pos.dtype,
    )
    flexion = joint_pos * signs
    left_force, right_force = _foot_contacts(env)
    contact_by_joint = {
        "Rthigh_to_Rleg": right_force > CONTACT_FORCE_THRESHOLD_N,
        "Lthigh_to_Lleg": left_force > CONTACT_FORCE_THRESHOLD_N,
    }
    contact = torch.stack([contact_by_joint[name] for name in names], dim=-1)
    stance_score = torch.exp(-torch.square(flexion - float(stance_target)) / (float(stance_sigma) ** 2))
    swing_score = torch.exp(-torch.square(flexion - float(swing_target)) / (float(swing_sigma) ** 2))
    score = torch.where(contact, stance_score, swing_score).mean(dim=-1)
    # 步行膝相位奖励不能影响静态 Stand；_masked 只给目标运动步态计分。
    return _masked(score, env, gait)


def foot_drag_penalty(env, gait: GaitMode | int, asset_cfg: SceneEntityCfg):
    """计算脚着地时仍发生水平滑动的拖脚成本。

    Args:
        env: 并行环境。
        gait: 本项对应的步态。
        asset_cfg: 按左脚、右脚选择足部刚体的配置。

    Returns:
        着地脚水平速度幅值之和 [m/s]，shape 为 ``[num_envs]``，其他步态为零。
    """

    foot_vel = _as_tensor(env.scene[asset_cfg.name].data.body_lin_vel_w)[:, asset_cfg.body_ids, :2]
    left, right = _foot_contacts(env)
    contact = torch.stack((left > CONTACT_FORCE_THRESHOLD_N, right > CONTACT_FORCE_THRESHOLD_N), dim=-1).float()
    value = torch.sum(torch.linalg.vector_norm(foot_vel, dim=-1) * contact, dim=-1)
    return _masked(value, env, gait)


def double_support_reward(env):
    """奖励 Stand 时左右脚同时有效接触地面。

    Returns:
        双脚支撑二值奖励，shape 为 ``[num_envs]``；非 Stand 环境为零。
    """

    left, right = _foot_contacts(env)
    score = ((left > CONTACT_FORCE_THRESHOLD_N) & (right > CONTACT_FORCE_THRESHOLD_N)).float()
    return _masked(score, env, GaitMode.STAND)


def standing_base_stillness_reward(env, sigma: float = 0.2):
    """奖励 Stand 时机身线速度和角速度接近零。

    Args:
        env: 并行环境。
        sigma: 合成速度平方的指数衰减尺度。

    Returns:
        静止相似度，shape 为 ``[num_envs]``，非 Stand 环境为零。
    """

    lin, ang, _ = _base(env)
    motion = torch.sum(torch.square(lin), dim=-1) + torch.sum(torch.square(ang), dim=-1)
    return _masked(torch.exp(-motion / (sigma * sigma)), env, GaitMode.STAND)


def standing_pose_reward(env, sigma: float = 0.35):
    """奖励 Stand 时八关节保持默认姿态。

    Args:
        env: 并行环境。
        sigma: 关节角均方误差的尺度 [rad]。

    Returns:
        默认姿态相似度，shape 为 ``[num_envs]``，非 Stand 环境为零。
    """

    data = env.scene["robot"].data
    error = _canonical_joint_tensor(env, data.joint_pos) - _canonical_joint_tensor(env, data.default_joint_pos)
    score = torch.exp(-torch.mean(torch.square(error), dim=-1) / (sigma * sigma))
    return _masked(score, env, GaitMode.STAND)


def standing_upright_reward(env, sigma: float = 0.2):
    """奖励 Stand 时机身滚转和俯仰角接近水平。

    Args:
        env: 并行环境。
        sigma: 姿态角误差尺度 [rad]。

    Returns:
        直立相似度，shape 为 ``[num_envs]``，非 Stand 环境为零。
    """

    roll, pitch, _ = euler_xyz_from_quat(_base(env)[2])
    score = torch.exp(-(torch.square(roll) + torch.square(pitch)) / (sigma * sigma))
    return _masked(score, env, GaitMode.STAND)


def standing_foot_flatness_reward(env, asset_cfg: SceneEntityCfg, sigma: float = 0.2):
    """奖励 Stand 时双脚水平贴地，同时忽略脚的偏航方向。

    Args:
        env: 并行环境。
        asset_cfg: 选择左右脚刚体的配置。
        sigma: 去除 yaw 后的脚倾斜误差尺度。

    Returns:
        双脚平整度相似度，shape 为 ``[num_envs]``，非 Stand 环境为零。
    """

    foot_quat = _as_tensor(env.scene[asset_cfg.name].data.body_link_quat_w)[:, asset_cfg.body_ids]
    yaw = math_utils.yaw_quat(foot_quat.reshape(-1, 4)).reshape_as(foot_quat)
    tilt = math_utils.quat_mul(math_utils.quat_conjugate(yaw), foot_quat)
    error = torch.mean(torch.sum(torch.square(tilt[..., :3]), dim=-1), dim=-1)
    return _masked(torch.exp(-error / (sigma * sigma)), env, GaitMode.STAND)


def standing_feet_alignment_reward(env, asset_cfg: SceneEntityCfg, sigma: float = 0.2):
    """奖励 Stand 时双脚偏航方向与机身根节点一致。

    Args:
        env: 并行环境。
        asset_cfg: 选择机器人和左右脚刚体的配置。
        sigma: 相对偏航误差尺度。

    Returns:
        双脚朝向相似度，shape 为 ``[num_envs]``，非 Stand 环境为零。
    """

    foot_quat = _as_tensor(env.scene[asset_cfg.name].data.body_link_quat_w)[:, asset_cfg.body_ids]
    root_quat = _as_tensor(env.scene[asset_cfg.name].data.root_quat_w)
    desired_yaw = math_utils.yaw_quat(root_quat).unsqueeze(1).expand_as(foot_quat)
    relative = math_utils.quat_mul(math_utils.quat_conjugate(desired_yaw), math_utils.yaw_quat(foot_quat))
    error = torch.mean(torch.sum(torch.square(relative[..., :3]), dim=-1), dim=-1)
    return _masked(torch.exp(-error / (sigma * sigma)), env, GaitMode.STAND)


def standing_joint_stillness_reward(env, sigma: float = 0.25):
    """奖励 Stand 时八关节角速度接近零。

    Args:
        env: 并行环境。
        sigma: 关节角速度均方误差尺度 [rad/s]。

    Returns:
        关节静止相似度，shape 为 ``[num_envs]``，非 Stand 环境为零。
    """

    joint_vel = _canonical_joint_tensor(env, env.scene["robot"].data.joint_vel)
    score = torch.exp(-torch.mean(torch.square(joint_vel), dim=-1) / (sigma * sigma))
    return _masked(score, env, GaitMode.STAND)


def smooth_slowdown_reward(env, gait: GaitMode | int, scale: float = 0.2):
    """奖励过渡步态中相邻状态的平面速度平稳下降。

    Args:
        env: 并行环境。
        gait: 应激活该奖励的过渡步态。
        scale: 速度下降映射到满分的尺度 [m/s]。

    Returns:
        ``[0, 1]`` 的减速分数，shape 为 ``[num_envs]``，其他步态为零。
    """

    current = torch.linalg.vector_norm(_base(env)[0][:, :2], dim=-1)
    previous = _gait_command(env).previous_speed
    score = torch.clamp((previous - current) / scale, min=0.0, max=1.0)
    return _masked(score, env, gait)


def short_contact_reward(env, max_contact_time: float = 0.25):
    """奖励 Run 中较短的单脚接触持续时间。

    Args:
        env: 并行环境。
        max_contact_time: 接触时间指数衰减尺度 [s]。

    Returns:
        当前着地脚的短接触平均分，shape 为 ``[num_envs]``；非 Run 环境为零。
        接触传感器关闭时直接返回全零。
    """

    if not _optional_sensor_enabled(env, "contact"):
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)

    durations = []
    for sensor_name in ("left_foot_contact", "right_foot_contact"):
        duration = (
            _as_tensor(env.scene[sensor_name].data.current_contact_time).reshape(env.num_envs, -1).max(dim=-1).values
        )
        durations.append(duration)
    contact_time = torch.stack(durations, dim=-1)
    in_contact = contact_time > 0.0
    score = (torch.exp(-contact_time / max_contact_time) * in_contact.float()).sum(dim=-1)
    score /= in_contact.sum(dim=-1).clamp(min=1)
    return _masked(score, env, GaitMode.RUN)


def push_off_reward(env, asset_cfg: SceneEntityCfg):
    """奖励单脚支撑时踝关节功率带来的向前/向上蹬伸。

    Args:
        env: 并行环境。
        asset_cfg: 关节必须按右踝、左踝排列的实体配置。

    Returns:
        踝功率与机身正向/向上运动的有界乘积，shape 为 ``[num_envs]``；只在 Run
        且恰好单脚支撑时非零。
    """

    robot = env.scene[asset_cfg.name]
    ankle_speed = torch.abs(_as_tensor(robot.data.joint_vel)[:, asset_cfg.joint_ids])
    effort = getattr(robot.data, "applied_torque", None)
    if effort is None:
        effort = getattr(robot.data, "computed_torque", None)
    if effort is None:
        effort = torch.zeros_like(_as_tensor(robot.data.joint_vel))
    ankle_effort = torch.abs(_as_tensor(effort)[:, asset_cfg.joint_ids])
    left, right = _foot_contacts(env)
    # 关节下标顺序固定为 [右踝, 左踝]，接触张量必须使用同一顺序。
    stance = torch.stack((right > CONTACT_FORCE_THRESHOLD_N, left > CONTACT_FORCE_THRESHOLD_N), dim=-1).float()
    single_support = (stance.sum(dim=-1) == 1.0).float()
    ankle_power = torch.sum(ankle_speed * ankle_effort * stance, dim=-1) / stance.sum(dim=-1).clamp(min=1.0)
    root_lin_vel = _base(env)[0]
    com_motion = torch.clamp(root_lin_vel[:, 0], min=0.0) + 0.5 * torch.clamp(root_lin_vel[:, 2], min=0.0)
    score = torch.tanh(ankle_power / 25.0) * torch.tanh(com_motion / 0.5) * single_support
    return _masked(score, env, GaitMode.RUN)


def forward_velocity_reward(env, gait: GaitMode | int, scale: float = 4.0):
    """奖励指定步态中的正向速度，并把结果限制到 ``[0, 1]``。

    Args:
        env: 并行环境。
        gait: 应激活本奖励的步态。
        scale: 达到满分的正向速度 [m/s]。

    Returns:
        有界正向速度分数，shape 为 ``[num_envs]``。
    """

    score = torch.clamp(_base(env)[0][:, 0] / scale, min=0.0, max=1.0)
    return _masked(score, env, gait)


def transition_to_walk_speed_reward(env, target_speed: float = 0.2, std: float = 0.3):
    """奖励 Run-to-Walk 过渡收敛到目标走路速度。

    Args:
        env: 并行环境。
        target_speed: 过渡结束时的正向目标速度 [m/s]。
        std: 速度误差指数核尺度 [m/s]。

    Returns:
        速度相似度，shape 为 ``[num_envs]``；非 Run-to-Walk 环境为零。
    """

    error = torch.square(_base(env)[0][:, 0] - target_speed)
    return _masked(torch.exp(-error / (std * std)), env, GaitMode.RUN_TO_WALK)


def advance_exoskeleton_curriculum(
    env,
    env_ids,
    phase2_start: int,
    phase3_start: int,
    walk_speed: float,
    transition_max_speed: float,
    multi_gait_max_speed: float,
    step_offset: int = 0,
):
    """按全局控制步推进 Walk、Transitions 和 Multi-gait 三阶段课程。

    Args:
        env: 带 ``common_step_counter`` 和步态命令项的环境。
        env_ids: Isaac Lab 课程管理器传入的环境下标；阶段是全局的，因此不使用。
        phase2_start: 进入 Transitions 阶段的全局控制步。
        phase3_start: 进入 Multi-gait 阶段的全局控制步。
        walk_speed: 第一阶段及后续阶段的最小前进速度 [m/s]。
        transition_max_speed: 第二阶段最大前进速度 [m/s]。
        multi_gait_max_speed: 第三阶段最大前进速度 [m/s]。
        step_offset: 恢复训练时补回的已完成控制步，必须非负。

    Returns:
        TensorBoard 可记录的阶段、控制步、当前最大命令速度和 Froude 阈值字典。

    Notes:
        RL-Games checkpoint 不序列化 ``env.common_step_counter``，因此从旧 checkpoint
        恢复时必须显式设置 ``step_offset``，否则课程会错误地从第一阶段重新开始。
    """

    del env_ids
    if step_offset < 0:
        raise ValueError("curriculum step_offset must be non-negative")
    curriculum_step = env.common_step_counter + step_offset
    phase = curriculum_phase_for_step(curriculum_step, phase2_start, phase3_start)
    command = _gait_command(env)
    if phase == CurriculumPhase.WALK_ONLY:
        command.cfg.ranges.lin_vel_x = (walk_speed, walk_speed)
        command.cfg.ranges.lin_vel_y = (0.0, 0.0)
        command.cfg.ranges.ang_vel_z = (0.0, 0.0)
        command.cfg.rel_standing_envs = 0.0
        max_speed = walk_speed
    elif phase == CurriculumPhase.TRANSITIONS:
        command.cfg.ranges.lin_vel_x = (walk_speed, transition_max_speed)
        command.cfg.ranges.lin_vel_y = (-0.15, 0.15)
        command.cfg.ranges.ang_vel_z = (-0.3, 0.3)
        command.cfg.rel_standing_envs = 0.2
        max_speed = transition_max_speed
    else:
        command.cfg.ranges.lin_vel_x = (walk_speed, multi_gait_max_speed)
        command.cfg.ranges.lin_vel_y = (-0.3, 0.3)
        command.cfg.ranges.ang_vel_z = (-0.5, 0.5)
        command.cfg.rel_standing_envs = 0.1
        max_speed = multi_gait_max_speed

    command.set_curriculum_phase(phase)
    return {
        "phase": float(phase),
        "control_step": float(curriculum_step),
        "max_command_speed_m_s": float(max_speed),
        "froude_threshold": FROUDE_THRESHOLD,
    }
