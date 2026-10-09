"""多模态外骨骼模仿任务的运行时环境。

基础行走逻辑在 my_exoskeleton_flat.py 中，本文件只增加“老师数据”这一层：从
ExoskeletonImitationDataset 为每个并行环境抽取任务片段，把当前时刻的关节、IMU、
RGB 和事件相机目标写入缓存，再推进序列游标。奖励函数随后从同一缓存读取目标，所以
“抽数据”和“算相似度”不会各自选择不同时间点。

主要数据流为：数据集窗口 -> 每环境目标缓存 -> 观测/模仿奖励 -> step 后游标前进。
常见 shape：关节 [num_envs, 8]、IMU [num_envs, 6, 9]、RGB 紧凑特征
[num_envs, 12]、事件特征 [num_envs, 32]。原始 256x256 图像不会直接拼入 PPO 向量，
否则数千个并行环境会消耗不可接受的显存。

阅读时先看 MyExoskeletonImitationEnv 的 __init__、_sample_targets、
_write_current_targets、_advance_target_sequences 和 step。以下以下划线开头的小函数
主要负责把原始媒体压缩为固定长度特征。
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_assets.envs.my_exoskeleton_imitation_env_cfg import MyExoskeletonImitationEnvCfg
from isaaclab_assets.tasks.imitation_dataset import ExoskeletonImitationDataset, resolve_imu_mapping
from isaaclab_assets.tasks.imitation_reward import (
    EVENT_FEATURES,
    IMU_COUNT,
    IMU_FEATURES,
    RGB_FEATURES,
    imitation_total_reward,
)
from isaaclab_assets.tasks.ruleresult import RuleresultRecorderMixin
from isaaclab_assets.tasks.vision_sensor_cfg import EVENT_CAMERA_NAMES, make_dvs_runtime

_LOGGER = logging.getLogger(__name__)

# 当某个 trial 没有经过标定的根部位移时，用下面的保守速度作为命令目标。它只提供
# “大致应该前进多快”，真正的模仿教师信号仍主要来自关节和 IMU 通道。
TASK_SPEED_M_S = {
    "BT01": 0.0,
    "BT02": 0.55,
    "BT03": 0.30,
    "BT04": 1.00,
    "BT05": 0.45,
    "AT01": 0.35,
    "AT02": 0.35,
    "AT03": 0.30,
    "AT04": 0.25,
    "AT05": 0.35,
}


# -----------------------------------------------------------------------------
# 多模态特征压缩与诊断摘要
# -----------------------------------------------------------------------------


def _event_feature(histogram: torch.Tensor) -> torch.Tensor:
    """把事件相机直方图压缩为固定 32 维特征。

    Args:
        histogram: 数据集窗口 ``[T, 2相机, 2极性, H, W]``，或已经合并时间的
            ``[2, 2, H, W]``。兼容旧数据的 ``[2极性, H, W]``。

    Returns:
        非负、经过 ``log1p`` 压缩的向量，shape 为 ``[32]``。shape 无法识别时
        返回全零，保证 PPO 输入维度不变。

    Notes:
        32 维来自四个“相机/极性”平面各自的 ``2x4`` 空间池化。
    """

    # 数据集窗口是 [T, 2相机, 2极性, H, W]；运行时适配器可能已合并时间维，
    # 直接给出 [2, 2, H, W]。
    if histogram.ndim == 5:
        histogram = histogram.sum(dim=0)
    if histogram.ndim == 4 and histogram.shape[0:2] == (2, 2):
        histogram = histogram.reshape(4, *histogram.shape[-2:])
    elif histogram.ndim == 3 and histogram.shape[0] == 2:
        # 兼容旧版把两个相机合并、只保留正负极性的录制。
        histogram = torch.cat((histogram, torch.zeros_like(histogram)), dim=0)
    if histogram.ndim != 3 or histogram.shape[0] != 4:
        return torch.zeros((EVENT_FEATURES,), dtype=torch.float32)
    histogram = histogram.float().clamp_min(0.0)
    # 四个相机/极性平面，每个平面保留八个空间统计格。
    pooled = torch.nn.functional.adaptive_avg_pool2d(histogram.unsqueeze(1), (2, 4))
    return torch.log1p(pooled.reshape(-1)).to(dtype=torch.float32)


def _rgb_feature(rgb: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, bool]:
    """把一个双目 RGB 时间窗口压缩为均值和标准差特征。

    Args:
        rgb: RGB 窗口，shape 为 ``[T, 2相机, 3通道, H, W]``。
        valid: 可选有效掩码；只有所有元素有效时输出标记才为真。

    Returns:
        ``(特征, 是否有效)``。特征对时间先取均值，再为每个相机/颜色通道计算
        空间均值和标准差，shape 为 ``[12]``。输入 shape 错误时返回全零和 ``False``。
    """

    if rgb.ndim != 5 or rgb.shape[1:3] != (2, 3):
        return torch.zeros((RGB_FEATURES,), dtype=torch.float32), False
    value = rgb.float()
    if value.numel() and float(value.detach().amax()) > 1.5:
        value = value / 255.0
    value = value.mean(dim=0)
    feature = torch.cat((value.mean(dim=(-1, -2)).reshape(-1), value.std(dim=(-1, -2), unbiased=False).reshape(-1)))
    available = True if valid is None else bool(torch.as_tensor(valid).bool().all())
    return feature.to(dtype=torch.float32), bool(available and torch.isfinite(feature).all())


def _rgb_feature_sequence(rgb: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, bool]:
    """逐帧把双目 RGB 序列压缩为 12 维统计特征。

    Args:
        rgb: shape 为 ``[T, 2, 3, H, W]`` 的 RGB 序列。
        valid: 可选帧有效掩码。

    Returns:
        ``(特征序列, 整窗是否有效)``，特征 shape 为 ``[T, 12]``。与
        :func:`_rgb_feature` 不同，本函数保留时间维供每个控制步读取。
    """

    if rgb.ndim != 5 or rgb.shape[1:3] != (2, 3):
        return torch.zeros((max(1, int(rgb.shape[0]) if rgb.ndim else 1), RGB_FEATURES)), False
    value = rgb.float()
    if value.numel() and float(value.detach().amax()) > 1.5:
        value = value / 255.0
    means = value.mean(dim=(-1, -2))
    stds = value.std(dim=(-1, -2), unbiased=False)
    feature = torch.cat((means[:, 0], means[:, 1], stds[:, 0], stds[:, 1]), dim=-1)
    available = True if valid is None else bool(torch.as_tensor(valid).bool().all())
    return feature.to(dtype=torch.float32), bool(available and torch.isfinite(feature).all())


def _event_feature_sequence(histogram: torch.Tensor) -> torch.Tensor:
    """逐帧压缩事件直方图，不混合相邻时间片。

    Args:
        histogram: shape 为 ``[T, 2相机, 2极性, H, W]`` 的事件序列。

    Returns:
        shape 为 ``[T, 32]`` 的特征序列；输入 shape 错误时返回 ``[1, 32]`` 全零。
    """

    if histogram.ndim != 5:
        return torch.zeros((1, EVENT_FEATURES), dtype=torch.float32)
    return torch.stack([_event_feature(frame) for frame in histogram], dim=0)


def _event_mask_batch(value: Any, num_envs: int, device: torch.device | str) -> torch.Tensor | None:
    """把一个 EVIS 极性掩码统一为 ``[num_envs, H, W]``。

    不同 EVIS 版本可能返回所有环境共用的 ``[H, W]``、批量 ``[N, H, W]``，或
    带单例通道的四维形式。显式规范 shape，可避免 4096 个克隆场景中把单环境维度
    错当成批量维。

    Args:
        value: EVIS 返回的数组或张量。
        num_envs: 期望批量大小 ``N``。
        device: 输出张量设备。

    Returns:
        ``float32`` 掩码 ``[N, H, W]``；无法识别或转换失败时返回 ``None``。
    """

    try:
        mask = torch.as_tensor(value, device=device, dtype=torch.float32)
    except (RuntimeError, TypeError, ValueError):
        return None
    if mask.ndim == 2:
        return mask.unsqueeze(0).expand(num_envs, -1, -1)
    if mask.ndim == 3:
        if mask.shape[0] == num_envs:
            return mask
        if mask.shape[0] == 1:
            return mask.expand(num_envs, -1, -1)
        return None
    if mask.ndim == 4:
        # 常见四维形式是 [N, 1, H, W] 和 [1, N, H, W]。
        if mask.shape[0] == num_envs and mask.shape[1] == 1:
            return mask[:, 0]
        if mask.shape[0] == 1 and mask.shape[1] == num_envs:
            return mask[0]
    return None


def _diagnostic_tensor_stats(value: Any) -> dict[str, Any] | None:
    """把任意张量诊断值压缩为可写入 JSON 的统计摘要。

    Args:
        value: 张量、数组、标量或 ``None``。

    Returns:
        包含 shape、元素数、有限值比例、均值和范围的字典；无法转换或输入为
        ``None`` 时返回 ``None``。原始大张量不会被序列化。
    """

    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value).detach().float()
    except (RuntimeError, TypeError, ValueError):
        return None
    if tensor.numel() == 0:
        return {"shape": list(tensor.shape), "count": 0}
    finite = torch.isfinite(tensor)
    if not bool(finite.any()):
        return {"shape": list(tensor.shape), "count": int(tensor.numel()), "finite_fraction": 0.0}
    finite_values = tensor[finite]
    return {
        "shape": list(tensor.shape),
        "count": int(tensor.numel()),
        "finite_fraction": float(finite.float().mean().item()),
        "mean": float(finite_values.mean().item()),
        "abs_mean": float(finite_values.abs().mean().item()),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
    }


def build_imitation_diagnostics(env: Any, reward: torch.Tensor | None = None) -> dict[str, Any]:
    """构造一次多模态回放状态的紧凑诊断快照。

    Args:
        env: 模仿环境或提供同名缓存属性的测试替身。
        reward: 可选逐环境模仿奖励，shape 为 ``[num_envs]``；不传时尝试读取奖励
            管理器最近缓冲区。

    Returns:
        只含 JSON 可序列化值的字典，包括任务计数、IMU/RGB/Event shape 与统计量、
        有效比例和奖励摘要。

    Notes:
        原始 RGB 帧和不等长事件不会写入 JSON，以免诊断文件巨大。环境仍把单环境
        RGB 原始张量保存在 ``_latest_camera_rgb``，供 Kit/Viser 交互查看。
    """

    imu = getattr(env, "_latest_imu", None)
    imu_valid = getattr(env, "_latest_imu_valid", None)
    rgb_feature = getattr(env, "_latest_rgb_feature", None)
    raw_rgb = getattr(env, "_latest_camera_rgb", {})
    event = getattr(env, "_latest_event_hist", None)
    event_valid = getattr(env, "_latest_event_valid", None)
    if reward is None:
        reward_manager = getattr(env, "reward_manager", None)
        reward = getattr(reward_manager, "_reward_buf", None)
    task_ids = getattr(env, "_imitation_task_ids_per_env", ())
    task_counts: dict[str, int] = {}
    for task_id in task_ids:
        key = str(task_id)
        task_counts[key] = task_counts.get(key, 0) + 1
    snapshot: dict[str, Any] = {
        "step": int(getattr(env, "_diagnostics_step", 0)),
        "time_s": float(getattr(getattr(env, "sim", None), "current_time", 0.0)),
        "num_envs": int(getattr(env, "num_envs", 0)),
        "tasks": task_counts,
        "imu": _diagnostic_tensor_stats(imu),
        "imu_valid_fraction": None
        if imu_valid is None
        else float(torch.as_tensor(imu_valid, dtype=torch.float32).mean().item()),
        "camera_rgb_features": _diagnostic_tensor_stats(rgb_feature),
        "camera_rgb_frames": {name: _diagnostic_tensor_stats(value) for name, value in raw_rgb.items()},
        "event_camera": _diagnostic_tensor_stats(event),
        "event_valid_fraction": None
        if event_valid is None
        else float(torch.as_tensor(event_valid, dtype=torch.float32).mean().item()),
        "imitation_reward": _diagnostic_tensor_stats(reward),
    }
    return snapshot


# -----------------------------------------------------------------------------
# 数据集条件模仿环境
# -----------------------------------------------------------------------------


class MyExoskeletonImitationEnv(RuleresultRecorderMixin, ManagerBasedRLEnv):
    """由多模态数据集提供逐环境教师目标的 Isaac Lab 环境。

    该类在基础平地环境之外管理五类时序缓存：IMU、关节运动、速度、RGB 特征和事件
    特征。每个并行环境有独立的任务 ID、有效标记、序列长度和游标，但相同任务的
    CPU 解码结果会共享，以避免数千环境重复解码视频。
    """

    cfg: MyExoskeletonImitationEnvCfg
    imitation_imu_sensor_names = ("imu_01", "imu_02", "imu_03", "imu_04", "imu_05", "imu_06")

    def sensor_enabled(self, name: str) -> bool:
        """读取配置中的可选传感器通道开关。

        Args:
            name: 通道短名称，例如 ``imu``、``rgb``、``event`` 或 ``contact``。

        Returns:
            ``use_<name>`` 的布尔值。旧配置和轻量测试替身没有新字段时默认启用，
            保持历史行为。
        """

        return bool(getattr(self.cfg, f"use_{name}", True))

    def _disable_sensor_entities(self, cfg: MyExoskeletonImitationEnvCfg | None = None) -> None:
        """在场景构造前移除配置中关闭的可选传感器实体。

        Args:
            cfg: 要修改的环境配置；不传时使用 ``self.cfg``。

        Notes:
            通道关闭后，观测项仍返回固定 shape 的全零张量，维持 PPO/checkpoint
            契约。本方法同时从场景配置删除实体，避免消融实验仍分配 RTX、DVS 或
            IMU 缓冲区。它必须早于 ``ManagerBasedRLEnv`` 创建场景执行。
        """

        if cfg is None:
            cfg = self.cfg
        scene_cfg = cfg.scene
        if not bool(getattr(cfg, "use_imu", True)):
            for name in (
                *self.imitation_imu_sensor_names,
                "imu_rthigh",
                "imu_rleg",
                "imu_rfoot",
                "imu_lthigh",
                "imu_lleg",
                "imu_lfoot",
            ):
                setattr(scene_cfg, name, None)
        if not bool(getattr(cfg, "use_rgb", True)):
            for name in ("left_rgb_camera", "right_rgb_camera"):
                setattr(scene_cfg, name, None)
        if not bool(getattr(cfg, "use_event", True)):
            for name in EVENT_CAMERA_NAMES:
                setattr(scene_cfg, name, None)
        if not bool(getattr(cfg, "use_contact", True)):
            for name in ("left_foot_contact", "right_foot_contact"):
                setattr(scene_cfg, name, None)

    def __init__(self, cfg: MyExoskeletonImitationEnvCfg, render_mode: str | None = None, **kwargs):
        """创建传感器缓存、基础场景、数据集和每环境教师目标。

        Args:
            cfg: 多模态模仿环境配置。
            render_mode: Gym/Isaac Lab 渲染模式；``None`` 使用配置默认值。
            **kwargs: 原样传给 :class:`ManagerBasedRLEnv` 的附加参数。

        Notes:
            父类构造期间就可能计算观测，因此所有固定 shape 缓存必须先创建。之后
            依次构造场景、解析 IMU 映射、加载数据集、连接可选 EVIS 运行时，最后
            为全部环境采样首个目标窗口。
        """

        # ManagerBasedRLEnv.__init__ 期间就会计算观测，所以必须先初始化其会读取的
        # 缓存，再把控制权交给 Isaac Lab。
        self._imitation_targets: dict[str, torch.Tensor] = {}
        self._dataset: ExoskeletonImitationDataset | None = None
        self._target_by_task: dict[str, dict[str, Any]] = {}
        self._imitation_task_ids: list[str] = []
        self._imitation_sequence_targets: dict[str, torch.Tensor] = {}
        self._imitation_sequence_lengths = torch.ones((cfg.scene.num_envs,), dtype=torch.long, device=cfg.sim.device)
        self._imitation_target_cursor = torch.zeros((cfg.scene.num_envs,), dtype=torch.long, device=cfg.sim.device)
        self._imitation_sequence_loop_per_env = torch.full(
            (cfg.scene.num_envs,),
            bool(getattr(cfg, "imitation_sequence_loop", True)),
            dtype=torch.bool,
            device=cfg.sim.device,
        )
        self._dvs_runtime: Any | None = None
        self._dvs_previous: Any | None = None
        self._event_last_process_time: float | None = None
        self._latest_event_hist = torch.zeros((cfg.scene.num_envs, EVENT_FEATURES), device=cfg.sim.device)
        self._latest_event_valid = torch.zeros((cfg.scene.num_envs,), dtype=torch.bool, device=cfg.sim.device)
        self._latest_camera_rgb: dict[str, torch.Tensor] = {}
        self._latest_rgb_feature = torch.zeros((cfg.scene.num_envs, RGB_FEATURES), device=cfg.sim.device)
        self._latest_imu = torch.zeros((cfg.scene.num_envs, IMU_COUNT, IMU_FEATURES), device=cfg.sim.device)
        self._latest_imu_valid = torch.zeros((cfg.scene.num_envs, IMU_COUNT), dtype=torch.bool, device=cfg.sim.device)
        self._diagnostics_step = 0
        self._diagnostics_stream: Any | None = None
        self._diagnostics_output_path: Path | None = None
        self._resolved_imu_links: tuple[str, ...] = (
            "Lfoot",
            "Rfoot",
            "Lleg",
            "Rleg",
            "Lthigh",
            "Rthigh",
        )
        self._imitation_task_ids_per_env: list[str] = []
        # CLI/Hydra 覆盖在构造环境前已生效；父类创建场景前移除关闭的传感器，确保
        # 它们确实不存在，而不只是被奖励函数忽略。
        self._disable_sensor_entities(cfg)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._resolve_imu_mapping()
        self._load_imitation_dataset()
        self._attach_event_runtime()
        self._sample_targets(None)

    @property
    def imitation_dataset(self) -> ExoskeletonImitationDataset | None:
        """返回已加载的数据集，供回放和 Viser 诊断；未配置时为 ``None``。"""

        return self._dataset

    def _resolve_imu_mapping(self) -> None:
        """把六个采集器 IMU 编号解析到实际导入的 articulation 刚体名。

        解析结果顺序为 ``imu_01`` 到 ``imu_06``，并写出一份 YAML 供人工核对。
        解析失败只记录警告并保留配置别名，因为某些轻量测试或旧场景没有刚体名称；
        正式训练应检查日志并确认六个链接都已解析。
        """

        if not self.sensor_enabled("imu"):
            _LOGGER.info("IMU channel disabled; skipping articulation IMU mapping")
            return

        try:
            robot = self.scene["robot"]
            body_names = tuple(str(name) for name in getattr(robot, "body_names", ()))
        except (AttributeError, KeyError, TypeError):
            body_names = ()
        if not body_names:
            _LOGGER.warning("robot body names are unavailable; using configured IMU link aliases")
            return

        output_path = getattr(self.cfg, "imu_mapping_output_path", None)
        output = Path(output_path).expanduser() if output_path else Path("imu_mapping_resolved.yaml")
        if not output.is_absolute():
            output = Path(self.cfg.log_dir or Path.cwd()) / output
        try:
            resolved = resolve_imu_mapping(body_names, output_path=output)
        except (OSError, ValueError, RuntimeError) as exc:
            _LOGGER.warning("unable to resolve IMU mapping against USD links: %s", exc)
            return
        defaults = self._resolved_imu_links
        self._resolved_imu_links = tuple(
            resolved.get(f"imu_{index:02d}") or defaults[index - 1] for index in range(1, IMU_COUNT + 1)
        )
        unresolved = [sensor_id for sensor_id, link in resolved.items() if link is None]
        if unresolved:
            _LOGGER.warning("unresolved IMU links %s; edit %s and update scene paths", unresolved, output)
        else:
            _LOGGER.info("resolved six IMU links against articulation: %s", self._resolved_imu_links)

    # -------------------------------------------------------------------------
    # 数据集、任务选择与教师序列缓存
    # -------------------------------------------------------------------------

    def _load_imitation_dataset(self) -> None:
        """根据配置加载多模态录制数据集并缓存可用任务 ID。

        数据路径优先使用 ``cfg.imitation_dataset_path``，为空时兼容命令配置中的目录
        ``reference_data_path``。没有路径表示只运行基础任务，不是错误。

        Raises:
            RuntimeError: 路径存在但数据集无法解析、任务无效或媒体读取初始化失败。
        """

        path = self.cfg.imitation_dataset_path
        command_cfg = self.cfg.commands.base_velocity
        reference_path = getattr(command_cfg, "reference_data_path", None)
        if not path and reference_path and Path(reference_path).is_dir():
            path = reference_path
        if not path:
            return
        task_id = str(self.cfg.imitation_task_id).upper()
        if task_id in ("", "ALL", "NONE"):
            task_id = getattr(command_cfg, "task_id", "all")
        task_id = str(task_id).upper()
        try:
            self._dataset = ExoskeletonImitationDataset(
                path,
                task_id=task_id,
                window_size=max(1, int(self.cfg.imitation_window_size)),
                target_dt_s=float(getattr(command_cfg, "reference_target_dt_s", 0.02)),
                random_window=True,
                random_task_sampling=bool(self.cfg.imitation_random_task_sampling),
                event_source_size=tuple(self.cfg.imitation_event_source_size),
                load_rgb=bool(self.sensor_enabled("rgb") and self.cfg.imitation_rgb_features),
                load_event=bool(self.sensor_enabled("event") and self.cfg.imitation_event_features),
                calibration_path=getattr(self.cfg, "imitation_calibration_path", None),
            )
            self._imitation_task_ids = list(self._dataset.task_ids)
            _LOGGER.info(
                "Loaded %d imitation trials (%s) from %s",
                len(self._dataset.trials),
                ", ".join(self._imitation_task_ids),
                path,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"unable to load imitation dataset '{path}': {exc}") from exc

    def _attach_event_runtime(self) -> None:
        """在可选 EVIS 插件存在时连接异步事件相机运行时。

        事件通道或事件特征关闭时直接跳过。标准 Isaac Sim 没有 EVIS 扩展时仅记录
        警告，并保持 ``event_valid=False``，这样 IMU/RGB 训练仍能继续。
        """

        try:
            if not self.sensor_enabled("event") or not self.cfg.imitation_event_features:
                _LOGGER.info("event channel disabled; skipping EVIS runtime")
                return
            output_dir = str(Path(self.cfg.log_dir or "/tmp") / "events")
            self._dvs_runtime = make_dvs_runtime(self.scene, EVENT_CAMERA_NAMES, out_dir=output_dir)
            for name in EVENT_CAMERA_NAMES:
                sensor = self.scene[name] if name in self.scene.keys() else None
                if sensor is not None and hasattr(sensor, "attach_runtime"):
                    sensor.attach_runtime(self._dvs_runtime)
        except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError) as exc:
            # 标准 Isaac Sim 通常没有 DVS 扩展。保留 IMU/RGB 训练能力，同时令
            # event_valid=False，并在日志明确暴露缺少的可选能力。
            self._dvs_runtime = None
            _LOGGER.warning("EVIS event runtime unavailable; event reward is masked: %s", exc)

    def _task_for_env(self, env_index: int) -> str:
        """在关闭随机任务采样时，为一个环境确定稳定任务 ID。

        Args:
            env_index: 并行环境的零基下标。

        Returns:
            配置明确选择且数据集存在的任务；否则按环境下标循环分配可用任务。
            未加载数据集时使用兼容默认值 ``BT02``。
        """

        if not self._imitation_task_ids:
            return "BT02"
        selected = str(self.cfg.imitation_task_id).upper()
        if selected in ("", "ALL", "NONE"):
            selected = str(getattr(self.cfg.commands.base_velocity, "task_id", "all")).upper()
        if selected in self._imitation_task_ids:
            return selected
        return self._imitation_task_ids[env_index % len(self._imitation_task_ids)]

    def _select_task_ids(self, ids: torch.Tensor) -> list[str]:
        """为每个待重置环境选择一个有效数据集任务。

        Args:
            ids: 环境下标张量，shape 为 ``[K]``。

        Returns:
            长度为 ``K`` 的任务 ID 列表。启用随机采样时独立随机选择，否则使用
            :meth:`_task_for_env` 的确定性分配。
        """

        if not self._imitation_task_ids:
            return ["BT02" for _ in range(ids.numel())]
        if self.cfg.imitation_random_task_sampling and len(self._imitation_task_ids) > 1:
            choices = torch.randint(len(self._imitation_task_ids), (ids.numel(),), device=self.device, dtype=torch.long)
            return [self._imitation_task_ids[int(index)] for index in choices.cpu().tolist()]
        return [self._task_for_env(int(index)) for index in ids.cpu().tolist()]

    def _target_for_task(self, task_id: str) -> dict[str, Any]:
        """取得某任务的一段同步教师序列，并按任务在 CPU 侧缓存。

        Args:
            task_id: 数据集任务编号，例如 ``BT02``。

        Returns:
            包含以下主要字段的字典，其中 ``T`` 是窗口长度：IMU
            ``target_imu [T, 6, 9]``、关节 ``target_motion [T, 8]``、速度
            ``target_velocity [T, 2]``、RGB ``[T, 12]``、Event ``[T, 32]``，以及
            各通道有效标记、任务 ID 和 ``sequence_length``。没有数据集时返回长度 1
            的全零教师目标，但保留任务速度目标。

        Notes:
            同一 reset 批次中相同任务只解码一次，避免数千环境重复读取视频。
            禁用的传感器通道保留原 shape，但数值清零且有效标记为假。
        """

        if task_id in self._target_by_task:
            return self._target_by_task[task_id]
        if self._dataset is None:
            value = {
                "target_imu": torch.zeros((1, IMU_COUNT, IMU_FEATURES)),
                "imu_valid": torch.zeros((IMU_COUNT,), dtype=torch.bool),
                "target_motion": torch.zeros((1, 8)),
                "motion_valid": torch.tensor(False),
                "target_velocity": torch.tensor(((TASK_SPEED_M_S.get(task_id, 0.3), 0.0),)),
                "velocity_valid": torch.tensor(True),
                "target_rgb_feature": torch.zeros((1, RGB_FEATURES)),
                "rgb_valid": torch.tensor(False),
                "target_event_feature": torch.zeros((1, EVENT_FEATURES)),
                "event_valid": torch.tensor(False),
                "task_id": task_id,
            }
            if not self.sensor_enabled("imu"):
                value["target_imu"].zero_()
                value["imu_valid"].zero_()
            if not self.sensor_enabled("rgb"):
                value["target_rgb_feature"].zero_()
                value["rgb_valid"] = torch.tensor(False)
            if not self.sensor_enabled("event"):
                value["target_event_feature"].zero_()
                value["event_valid"] = torch.tensor(False)
            value["sequence_length"] = 1
            self._target_by_task[task_id] = value
            return value
        sample = self._dataset.sample(
            task_id=task_id,
            window_size=max(1, int(self.cfg.imitation_window_size)),
            random_window=True,
        )
        rgb_feature, rgb_valid = _rgb_feature_sequence(sample["target_rgb"], sample.get("rgb_valid"))
        event_feature = _event_feature_sequence(sample["target_event"])
        # 安静场景的 DVS 直方图为零，但仍是有效观测；有效性取决于能否成功解码，
        # 不要求两个相机在每个 20 ms 窗口都必须产生事件。
        event_valid = bool(sample["event_valid"].item())
        speed = TASK_SPEED_M_S.get(task_id, 0.3)
        value = {
            "target_imu": sample["target_imu"].float(),
            "imu_valid": sample["imu_valid"].bool(),
            "target_motion": sample["target_motion"].float(),
            "motion_valid": sample["motion_valid"].bool(),
            "target_velocity": torch.tensor((speed, 0.0), dtype=torch.float32).expand(sample["target_imu"].shape[0], 2),
            # 即使录制缺少根节点位移，任务速度也是校准后的回退目标。把它标记为
            # 有效，使每个同步窗口都能计算速度相似度。
            "velocity_valid": torch.tensor(True),
            "target_rgb_feature": rgb_feature,
            "rgb_valid": torch.tensor(rgb_valid),
            "target_event_feature": event_feature,
            "event_valid": torch.tensor(event_valid),
            "task_id": task_id,
            "sequence_length": int(sample["target_imu"].shape[0]),
        }
        # 屏蔽关闭通道时仍保留时间缓冲 shape，使纯状态传感器消融不改变 PPO 观测
        # 维度和 checkpoint 契约。
        if not self.sensor_enabled("imu"):
            value["target_imu"].zero_()
            value["imu_valid"] = torch.zeros_like(value["imu_valid"], dtype=torch.bool)
        if not self.sensor_enabled("rgb"):
            value["target_rgb_feature"].zero_()
            value["rgb_valid"] = torch.tensor(False)
        if not self.sensor_enabled("event"):
            value["target_event_feature"].zero_()
            value["event_valid"] = torch.tensor(False)
        self._target_by_task[task_id] = value
        return value

    def _apply_task_speed_targets(self, env_ids: torch.Tensor, task_ids: Sequence[str]) -> None:
        """把数据集任务对应的前进速度写入逐环境命令项。

        Args:
            env_ids: 环境下标，shape 为 ``[K]``。
            task_ids: 与下标一一对应的 ``K`` 个任务 ID。

        Notes:
            正式 :class:`GaitVelocityCommand` 通过 ``set_task_speed`` 同步步态状态；
            轻量测试替身没有该方法时，才直接写其 ``command`` 张量。
        """

        if env_ids.numel() == 0:
            return
        try:
            command = self.command_manager.get_term("base_velocity")
        except (AttributeError, KeyError, TypeError):
            return
        speeds = torch.as_tensor(
            [TASK_SPEED_M_S.get(str(task_id).upper(), 0.3) for task_id in task_ids],
            device=self.device,
            dtype=torch.float32,
        )
        setter = getattr(command, "set_task_speed", None)
        if callable(setter):
            setter(env_ids, speeds)
            return
        # 保持轻量测试替身和自定义命令项可用；正式 GaitVelocityCommand 会走上面的
        # setter 路径，回退分支只在存在公开命令张量时直接写入。
        command_value = getattr(command, "command", None)
        if command_value is not None:
            command_value[env_ids, 0] = speeds
            command_value[env_ids, 1:] = 0.0

    def _sample_targets(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        """为选中环境采样任务窗口，并安装 GPU 教师序列与当前帧。

        Args:
            env_ids: 要更新的环境下标 ``[K]``；``None`` 表示全部环境。

        数据流：
            1. 为每个环境选择任务 ID；
            2. 每个任务只在 CPU 解码一个同步窗口；
            3. 创建/扩展 ``[num_envs, max_length, ...]`` GPU 序列缓冲；
            4. 为循环任务随机设置独立起始相位，非循环任务到末帧后保持；
            5. 写入有效掩码、任务速度和当前游标对应的奖励目标。

        Notes:
            只重置部分环境时，共享缓冲不能缩短，否则仍处于旧 episode 的其他环境
            会被截断。因此 ``max_length`` 至少保持现有缓冲长度。
        """

        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        task_ids = self._select_task_ids(ids.detach())
        # 每个任务只在 CPU 解码一次，再把序列复制到 GPU；既避免 reset 时重复解码
        # 4096 次视频，又保留每个控制步的时序目标。
        self._target_by_task.clear()
        values = [self._target_for_task(task_id) for task_id in task_ids]

        max_length = max(int(value["sequence_length"]) for value in values)
        # 子集 reset 不能缩短其他环境仍在使用的共享缓冲；短的新窗口不应悄悄截断
        # 未结束环境的目标游标。
        if self._imitation_sequence_targets:
            max_length = max(max_length, next(iter(self._imitation_sequence_targets.values())).shape[1])
        sequence_shapes = {
            "target_imu": (IMU_COUNT, IMU_FEATURES),
            "target_motion": (8,),
            "target_velocity": (2,),
            "target_rgb_feature": (RGB_FEATURES,),
            "target_event_feature": (EVENT_FEATURES,),
        }
        needs_new_buffers = not self._imitation_sequence_targets or any(
            tensor.shape[1] != max_length for tensor in self._imitation_sequence_targets.values()
        )
        if needs_new_buffers:
            old_targets = self._imitation_sequence_targets
            self._imitation_sequence_targets = {
                name: torch.zeros((self.num_envs, max_length, *shape), device=self.device)
                for name, shape in sequence_shapes.items()
            }
            if old_targets:
                copy_length = min(max_length, next(iter(old_targets.values())).shape[1])
                for name, tensor in self._imitation_sequence_targets.items():
                    if name in old_targets:
                        tensor[:, :copy_length] = old_targets[name][:, :copy_length]

        loop_enabled = getattr(self, "_imitation_sequence_loop_per_env", None)
        if loop_enabled is None or loop_enabled.shape[0] != self.num_envs:
            loop_enabled = torch.full(
                (self.num_envs,),
                bool(getattr(self.cfg, "imitation_sequence_loop", True)),
                dtype=torch.bool,
                device=self.device,
            )
            self._imitation_sequence_loop_per_env = loop_enabled
        non_cyclic_tasks = {str(value).upper() for value in getattr(self.cfg, "imitation_non_cyclic_tasks", ("BT05",))}
        for env_index, value in zip(ids.cpu().tolist(), values, strict=True):
            length = int(value["sequence_length"])
            for name in sequence_shapes:
                sequence = value[name].to(device=self.device, dtype=torch.float32)
                self._imitation_sequence_targets[name][env_index, :length] = sequence
                if length < max_length:
                    self._imitation_sequence_targets[name][env_index, length:] = sequence[-1]
            self._imitation_sequence_lengths[env_index] = length
            # 同任务共享解码序列以控制 4096 环境 reset 成本；每环境独立随机循环
            # 相位，避免所有环境在同一控制步收到完全相同的教师帧。
            self._imitation_target_cursor[env_index] = int(torch.randint(max(1, length), (), device=self.device).item())

        if not self._imitation_targets or self._imitation_targets["target_imu"].shape[0] != self.num_envs:
            self._imitation_targets = {
                "target_imu": torch.zeros((self.num_envs, IMU_COUNT, IMU_FEATURES), device=self.device),
                "imu_valid": torch.zeros((self.num_envs, IMU_COUNT), dtype=torch.bool, device=self.device),
                "target_motion": torch.zeros((self.num_envs, 8), device=self.device),
                "motion_valid": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
                "target_velocity": torch.zeros((self.num_envs, 2), device=self.device),
                "velocity_valid": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
                "target_rgb_feature": torch.zeros((self.num_envs, RGB_FEATURES), device=self.device),
                "rgb_valid": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
                "target_event_feature": torch.zeros((self.num_envs, EVENT_FEATURES), device=self.device),
                "event_valid": torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            }
        for env_index, value in zip(ids.cpu().tolist(), values, strict=True):
            self._imitation_targets["imu_valid"][env_index] = value["imu_valid"].to(self.device)
            self._imitation_targets["motion_valid"][env_index] = value["motion_valid"].to(self.device)
            self._imitation_targets["velocity_valid"][env_index] = value["velocity_valid"].to(self.device)
            self._imitation_targets["rgb_valid"][env_index] = value["rgb_valid"].to(self.device)
            self._imitation_targets["event_valid"][env_index] = value["event_valid"].to(self.device)
        if len(self._imitation_task_ids_per_env) != self.num_envs:
            self._imitation_task_ids_per_env = ["BT02"] * self.num_envs
        for env_index, task_id in zip(ids.cpu().tolist(), task_ids, strict=True):
            self._imitation_task_ids_per_env[env_index] = task_id
            loop_enabled[env_index] = (
                bool(getattr(self.cfg, "imitation_sequence_loop", True))
                and str(task_id).upper() not in non_cyclic_tasks
            )
        # 轻量测试替身和只复用数据缓冲的调用方可能没有速度 hook；正式环境始终提供
        # 该方法，但缺少可选 hook 不应阻止安装 reset 教师目标。
        apply_speed_targets = getattr(self, "_apply_task_speed_targets", None)
        if callable(apply_speed_targets):
            apply_speed_targets(ids, task_ids)
        self._write_current_targets(ids)

    # -------------------------------------------------------------------------
    # 每控制步目标推进、诊断与资源释放
    # -------------------------------------------------------------------------

    def _write_current_targets(self, env_ids: torch.Tensor | None = None) -> None:
        """按每环境游标取出当前教师帧，写入奖励管理器读取的缓存。

        Args:
            env_ids: 要更新的环境下标 ``[K]``；``None`` 表示全部环境。

        Notes:
            序列缓存 shape 为 ``[num_envs, T, ...]``，游标 shape 为 ``[K]``。
            ``gather`` 为不同环境从各自时间位置取值，输出去掉时间维后写入
            ``_imitation_targets``，供本控制步的观测和奖励共同读取。
        """

        if not self._imitation_sequence_targets:
            return
        ids = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        )
        if ids.numel() == 0:
            return
        cursor = self._imitation_target_cursor[ids]
        for name, sequence in self._imitation_sequence_targets.items():
            index = cursor.view(-1, 1, *([1] * (sequence.ndim - 2))).expand(-1, 1, *sequence.shape[2:])
            self._imitation_targets[name][ids] = sequence[ids].gather(1, index).squeeze(1)

    def _advance_target_sequences(self, done: torch.Tensor | None = None) -> None:
        """把未终止环境的模仿目标推进一个 50 Hz 帧。

        Args:
            done: 本步已终止或截断的环境掩码，shape 为 ``[num_envs]``。这些环境
                不推进，因为 reset 路径会为它们采样新窗口。

        Notes:
            循环任务在末尾取模回到零；非循环任务（例如默认 BT05）保持最后一帧，
            防止动作序列完成后突然跳回起点。
        """

        if not self._imitation_sequence_targets:
            return
        active = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        if done is not None:
            active &= ~done.to(device=self.device, dtype=torch.bool).reshape(-1)
        lengths = self._imitation_sequence_lengths[active].clamp_min(1)
        next_cursor = self._imitation_target_cursor[active] + 1
        loop_enabled = getattr(self, "_imitation_sequence_loop_per_env", None)
        if loop_enabled is None:
            loop_enabled = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        loop_enabled = loop_enabled[active]
        wrapped = torch.remainder(next_cursor, lengths)
        held = torch.minimum(next_cursor, lengths - 1)
        self._imitation_target_cursor[active] = torch.where(loop_enabled, wrapped, held)
        self._write_current_targets(torch.where(active)[0])

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """重置指定环境，并同步刷新教师目标与事件相机状态。

        Args:
            env_ids: Isaac Lab 选中的环境下标。

        Notes:
            必须先让父类重置物理状态，再安装新教师窗口。事件运行时若支持逐环境
            reset 也同步调用；其失败只记调试日志，事件有效标记会清零。
        """

        super()._reset_idx(env_ids)
        self._sample_targets(env_ids)
        if self.sensor_enabled("event") and self._dvs_runtime is not None and hasattr(self._dvs_runtime, "reset"):
            try:
                self._dvs_runtime.reset(torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
            except (AttributeError, RuntimeError, TypeError) as exc:
                _LOGGER.debug("event runtime reset failed: %s", exc)
        self._dvs_previous = None
        self._event_last_process_time = None
        self._latest_event_valid.zero_()

    def step(self, action: torch.Tensor):
        """推进一个环境控制步，并在完成奖励计算后移动教师序列游标。

        Args:
            action: Actor 的归一化八关节动作，shape 为 ``[num_envs, 8]``。

        Returns:
            与 :meth:`ManagerBasedRLEnv.step` 相同的观测、奖励、终止、截断和信息。

        Notes:
            游标在 ``super().step`` 之后推进，因此本步奖励比较的是推进前的当前
            教师帧。已完成环境不推进，随后由 ``_reset_idx`` 安装新窗口。
        """

        result = super().step(action)
        self._ruleresult_record_after_step(result)
        self._diagnostics_step += 1
        self._emit_diagnostics_if_due()
        if isinstance(result, tuple) and len(result) >= 4:
            terminated, truncated = result[2], result[3]
            done = torch.as_tensor(terminated, device=self.device, dtype=torch.bool) | torch.as_tensor(
                truncated, device=self.device, dtype=torch.bool
            )
            self._advance_target_sequences(done)
        else:  # pragma: no cover - Isaac Lab 正常情况下始终返回五元素元组。
            self._advance_target_sequences()
        return result

    def _emit_diagnostics_if_due(self) -> None:
        """按配置间隔写出一次紧凑的多模态回放诊断。

        诊断只在 ``diagnostics_enabled`` 开启且步数达到间隔时执行。优先重新计算
        六通道模仿总分；失败时回退到奖励管理器缓冲。可选 JSONL 输出使用追加模式，
        每次立即 flush，便于训练/回放进行中查看。
        """

        if not bool(getattr(self.cfg, "diagnostics_enabled", False)):
            return
        interval = max(1, int(getattr(self.cfg, "diagnostics_interval", 20)))
        if self._diagnostics_step % interval:
            return
        try:
            reward = None
            try:
                # 只在稀疏诊断时刻重算，让快照报告六通道模仿分，而不是整个行走任务
                # 的奖励总和；避免每个控制步增加额外计算开销。
                with torch.no_grad():
                    reward = imitation_total_reward(self)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                reward = getattr(getattr(self, "reward_manager", None), "_reward_buf", None)
            snapshot = build_imitation_diagnostics(self, reward)
            output_path = getattr(self.cfg, "diagnostics_output_path", None)
            if output_path:
                path = Path(output_path).expanduser()
                if not path.is_absolute():
                    path = Path(getattr(self.cfg, "log_dir", None) or Path.cwd()) / path
                if self._diagnostics_stream is None or self._diagnostics_output_path != path:
                    if self._diagnostics_stream is not None:
                        self._diagnostics_stream.close()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self._diagnostics_stream = path.open("a", encoding="utf-8")
                    self._diagnostics_output_path = path
                self._diagnostics_stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
                self._diagnostics_stream.flush()
            _LOGGER.info(
                "imitation diagnostics step=%d reward=%s imu_valid=%s event_valid=%s",
                snapshot["step"],
                snapshot["imitation_reward"].get("mean") if snapshot["imitation_reward"] else None,
                snapshot["imu_valid_fraction"],
                snapshot["event_valid_fraction"],
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _LOGGER.warning("failed to emit imitation diagnostics: %s", exc)

    def diagnostics_snapshot(self) -> dict[str, Any]:
        """为外部查看器即时构造一份紧凑多模态快照。

        Returns:
            与 :func:`build_imitation_diagnostics` 相同的 JSON 安全字典。
        """

        reward = getattr(getattr(self, "reward_manager", None), "_reward_buf", None)
        return build_imitation_diagnostics(self, reward)

    def update_event_observation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """取得当前仿真步固定 shape 的事件相机特征。

        Returns:
            ``(事件特征, 有效掩码)``，shape 分别为 ``[num_envs, 32]`` 和
            ``[num_envs]``。通道关闭、插件不可用或尚无首个事件帧时返回全零特征和
            ``False`` 掩码。

        Notes:
            观测管理器和奖励管理器可能在同一仿真步各调用一次。本方法用仿真时间
            缓存结果，确保 EVIS 每步只推进一次，避免异步参考被重复推进和事件重复
            计数。
        """

        if not self.sensor_enabled("event") or self._dvs_runtime is None:
            zeros = torch.zeros((self.num_envs, EVENT_FEATURES), device=self.device)
            valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self._latest_event_hist = zeros
            self._latest_event_valid = valid
            return zeros, valid
        # 观测和奖励管理器可能在同一仿真步查询；EVIS 每步只处理一次，否则第二次
        # 查询会重复推进异步参考并重复计数事件。
        current_time = float(self.sim.current_time)
        if self._event_last_process_time is not None and current_time == self._event_last_process_time:
            return self._latest_event_hist, self._latest_event_valid
        try:
            if not hasattr(self._dvs_runtime, "process"):
                raise RuntimeError("installed EVIS runtime has no process() adapter")
            self._dvs_runtime.process(current_time)
            processors = tuple(getattr(self._dvs_runtime, "procs", ()))
            if len(processors) != len(EVENT_CAMERA_NAMES):
                raise RuntimeError("EVIS processor count does not match the two event cameras")
            planes: list[torch.Tensor] = []
            for processor in processors:
                masks = getattr(processor, "last_masks", None)
                if masks is None or len(masks) != 2:
                    # 第一帧只初始化 DVS 参考，没有事件掩码；在真实事件帧到来前保持
                    # 无效，不能把全零初始化误当有效静态场景。
                    self._event_last_process_time = current_time
                    self._latest_event_valid.zero_()
                    return self._latest_event_hist, self._latest_event_valid
                positive, negative = masks
                positive_batch = _event_mask_batch(positive, self.num_envs, self.device)
                negative_batch = _event_mask_batch(negative, self.num_envs, self.device)
                if positive_batch is None or negative_batch is None:
                    raise RuntimeError("EVIS polarity masks must be [H,W] or [N,H,W]")
                if positive_batch.shape != negative_batch.shape:
                    raise RuntimeError("EVIS positive and negative masks have different shapes")
                planes.extend((positive_batch, negative_batch))
            # 每个平面已经是 [N,H,W]；沿通道轴堆叠后，无论单环境还是 4096 个克隆
            # 环境，都稳定得到 [N,4,H,W]。
            stacked = torch.stack(planes, dim=1).to(device=self.device)
            if stacked.ndim != 4 or stacked.shape[0] != self.num_envs or stacked.shape[1] != 4:
                raise RuntimeError(f"EVIS masks produced invalid batch shape {tuple(stacked.shape)}")
            stacked = torch.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
            # [N,4,H,W] -> [N,4,2,4] -> 每环境 32 维紧凑事件特征。
            pooled = torch.nn.functional.adaptive_avg_pool2d(stacked, (2, 4))
            value = torch.log1p(pooled).reshape(self.num_envs, EVENT_FEATURES)
            self._latest_event_hist = value
            self._latest_event_valid = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
            self._event_last_process_time = current_time
        except Exception as exc:  # pragma: no cover - 具体异常依赖可选 EVIS 版本。
            _LOGGER.debug("event runtime did not return a fixed histogram: %s", exc)
            self._event_last_process_time = current_time
            self._latest_event_valid.zero_()
        return self._latest_event_hist, self._latest_event_valid

    def close(self):
        """关闭诊断文件、数据集、事件运行时和 Isaac Lab 环境。

        Returns:
            父类 :meth:`ManagerBasedRLEnv.close` 的返回值。

        Notes:
            清理按“本类资源 -> 父类场景”顺序执行。可选事件运行时关闭失败只记录
            调试日志，避免遮蔽其他资源的释放。
        """

        if self._diagnostics_stream is not None:
            with contextlib.suppress(OSError):
                self._diagnostics_stream.close()
            self._diagnostics_stream = None
        if self._dataset is not None:
            self._dataset.close()
        if self._dvs_runtime is not None and hasattr(self._dvs_runtime, "close"):
            try:
                self._dvs_runtime.close()
            except (AttributeError, RuntimeError, TypeError) as exc:
                _LOGGER.debug("event runtime close failed: %s", exc)
        return super().close()


# 为仍按旧名称导入任务环境的调用方保留兼容别名。
MyExoskeletonImitationRLEnv = MyExoskeletonImitationEnv

__all__ = [
    "MyExoskeletonImitationEnv",
    "MyExoskeletonImitationRLEnv",
    "TASK_SPEED_M_S",
    "build_imitation_diagnostics",
]
