"""Runtime environment for the multimodal exoskeleton imitation task."""

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

# Conservative command targets used when a trial has no calibrated root-motion
# file.  The IMU/joint channels remain the primary teacher signal.
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


def _event_feature(histogram: torch.Tensor) -> torch.Tensor:
    """Pool a time/camera/polarity histogram into 32 fixed features."""

    # Dataset windows are [T, 2 cameras, 2 polarities, H, W]. Runtime
    # adapters may already reduce time and provide [2, 2, H, W].
    if histogram.ndim == 5:
        histogram = histogram.sum(dim=0)
    if histogram.ndim == 4 and histogram.shape[0:2] == (2, 2):
        histogram = histogram.reshape(4, *histogram.shape[-2:])
    elif histogram.ndim == 3 and histogram.shape[0] == 2:
        # Compatibility with older merged two-polarity recordings.
        histogram = torch.cat((histogram, torch.zeros_like(histogram)), dim=0)
    if histogram.ndim != 3 or histogram.shape[0] != 4:
        return torch.zeros((EVENT_FEATURES,), dtype=torch.float32)
    histogram = histogram.float().clamp_min(0.0)
    # Four camera/polarity planes and eight spatial bins each.
    pooled = torch.nn.functional.adaptive_avg_pool2d(histogram.unsqueeze(1), (2, 4))
    return torch.log1p(pooled.reshape(-1)).to(dtype=torch.float32)


def _rgb_feature(rgb: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, bool]:
    """Reduce one sampled ``[T,2,3,H,W]`` RGB window to moments."""

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
    """Reduce ``[T,2,3,H,W]`` RGB frames to ``[T,12]`` moments."""

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
    """Pool ``[T,2,2,H,W]`` event histograms without mixing time bins."""

    if histogram.ndim != 5:
        return torch.zeros((1, EVENT_FEATURES), dtype=torch.float32)
    return torch.stack([_event_feature(frame) for frame in histogram], dim=0)


def _event_mask_batch(value: Any, num_envs: int, device: torch.device | str) -> torch.Tensor | None:
    """Normalize one EVIS polarity mask to ``[num_envs, height, width]``.

    EVIS versions expose ``last_masks`` either as one image shared by all
    environments (``[H, W]``) or as a batched tensor (``[N, H, W]``).  A few
    releases add a singleton channel dimension.  Keeping this conversion
    explicit prevents a single-environment mask from being mistaken for the
    environment axis when 4096 cloned scenes are used.
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
        # Common forms are [N, 1, H, W] and [1, N, H, W].
        if mask.shape[0] == num_envs and mask.shape[1] == 1:
            return mask[:, 0]
        if mask.shape[0] == 1 and mask.shape[1] == num_envs:
            return mask[0]
    return None


def _diagnostic_tensor_stats(value: Any) -> dict[str, Any] | None:
    """Return compact JSON-safe statistics for a tensor-like diagnostic value."""

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
    """Build a compact snapshot of the multimodal play-time state.

    Raw RGB frames and ragged event tuples are intentionally summarized rather
    than serialized.  The environment still keeps raw one-environment RGB
    tensors in ``_latest_camera_rgb`` for Kit/Viser inspection.
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


class MyExoskeletonImitationEnv(RuleresultRecorderMixin, ManagerBasedRLEnv):
    """Manager-based Isaac Lab environment with dataset-conditioned targets."""

    cfg: MyExoskeletonImitationEnvCfg
    imitation_imu_sensor_names = ("imu_01", "imu_02", "imu_03", "imu_04", "imu_05", "imu_06")

    def sensor_enabled(self, name: str) -> bool:
        """Return whether an optional sensor channel is enabled in the cfg.

        Older configs and dependency-light test doubles do not define the new
        fields, so the compatibility default is enabled.
        """

        return bool(getattr(self.cfg, f"use_{name}", True))

    def _disable_sensor_entities(self, cfg: MyExoskeletonImitationEnvCfg | None = None) -> None:
        """Remove disabled optional sensors before the scene is constructed.

        The observation terms keep their fixed shapes and return zero tensors
        when a channel is disabled.  Omitting the corresponding scene entity
        here additionally avoids allocating RTX/DVS/IMU buffers for an
        ablation run.  This method must run before ``ManagerBasedRLEnv``
        creates its :class:`InteractiveScene`.
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
        # Observation terms are evaluated during ManagerBasedRLEnv.__init__, so
        # initialize their caches before handing control to Isaac Lab.
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
        self._latest_imu_valid = torch.zeros(
            (cfg.scene.num_envs, IMU_COUNT), dtype=torch.bool, device=cfg.sim.device
        )
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
        # CLI/Hydra overrides are applied before the environment constructor.
        # Apply them before the base class builds the scene so disabled sensors
        # are genuinely absent rather than merely ignored by reward terms.
        self._disable_sensor_entities(cfg)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._resolve_imu_mapping()
        self._load_imitation_dataset()
        self._attach_event_runtime()
        self._sample_targets(None)

    @property
    def imitation_dataset(self) -> ExoskeletonImitationDataset | None:
        """The loaded dataset, exposed for play/viser diagnostics."""

        return self._dataset

    def _resolve_imu_mapping(self) -> None:
        """Resolve configured IMUs against imported articulation body names."""

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

    def _load_imitation_dataset(self) -> None:
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
        """Create EVIS' asynchronous runtime when the optional plugin exists."""

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
            # A stock Isaac Sim checkout has no DVS extension.  Keep the task
            # usable for IMU/RGB training, but leave event_valid=False and make
            # the missing capability visible in logs.
            self._dvs_runtime = None
            _LOGGER.warning("EVIS event runtime unavailable; event reward is masked: %s", exc)

    def _task_for_env(self, env_index: int) -> str:
        """Return the deterministic task used when random sampling is disabled."""

        if not self._imitation_task_ids:
            return "BT02"
        selected = str(self.cfg.imitation_task_id).upper()
        if selected in ("", "ALL", "NONE"):
            selected = str(getattr(self.cfg.commands.base_velocity, "task_id", "all")).upper()
        if selected in self._imitation_task_ids:
            return selected
        return self._imitation_task_ids[env_index % len(self._imitation_task_ids)]

    def _select_task_ids(self, ids: torch.Tensor) -> list[str]:
        """Select one valid dataset task per reset environment."""

        if not self._imitation_task_ids:
            return ["BT02" for _ in range(ids.numel())]
        if self.cfg.imitation_random_task_sampling and len(self._imitation_task_ids) > 1:
            choices = torch.randint(
                len(self._imitation_task_ids), (ids.numel(),), device=self.device, dtype=torch.long
            )
            return [self._imitation_task_ids[int(index)] for index in choices.cpu().tolist()]
        return [self._task_for_env(int(index)) for index in ids.cpu().tolist()]

    def _target_for_task(self, task_id: str) -> dict[str, Any]:
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
        # A quiet camera is a valid DVS observation (its histogram is zero),
        # so validity depends on successful decoding rather than requiring an
        # event from both cameras in every 20 ms window.
        event_valid = bool(sample["event_valid"].item())
        speed = TASK_SPEED_M_S.get(task_id, 0.3)
        value = {
            "target_imu": sample["target_imu"].float(),
            "imu_valid": sample["imu_valid"].bool(),
            "target_motion": sample["target_motion"].float(),
            "motion_valid": sample["motion_valid"].bool(),
            "target_velocity": torch.tensor((speed, 0.0), dtype=torch.float32).expand(sample["target_imu"].shape[0], 2),
            # The task speed is a calibrated fallback target even when the
            # recording has no root-position channel.  Mark it valid so the
            # velocity similarity remains active for every synchronized window.
            "velocity_valid": torch.tensor(True),
            "target_rgb_feature": rgb_feature,
            "rgb_valid": torch.tensor(rgb_valid),
            "target_event_feature": event_feature,
            "event_valid": torch.tensor(event_valid),
            "task_id": task_id,
            "sequence_length": int(sample["target_imu"].shape[0]),
        }
        # Keep the temporal buffer shape intact while masking disabled
        # channels. This allows state-only sensor ablations without changing
        # the PPO observation dimensions or checkpoint contract.
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
        """Apply dataset task speeds to the per-environment command term."""

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
        # Keep lightweight fakes and custom command terms usable.  The stock
        # GaitVelocityCommand takes the setter path above, while this fallback
        # only writes the documented command tensor when available.
        command_value = getattr(command, "command", None)
        if command_value is not None:
            command_value[env_ids, 0] = speeds
            command_value[env_ids, 1:] = 0.0

    def _sample_targets(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        task_ids = self._select_task_ids(ids.detach())
        # Build a per-task CPU cache once, then copy sequences to GPU.  This
        # avoids decoding a video 4096 times at reset while retaining temporal
        # targets for every control step.
        self._target_by_task.clear()
        values = [self._target_for_task(task_id) for task_id in task_ids]

        max_length = max(int(value["sequence_length"]) for value in values)
        # Resetting a subset must not shrink the shared buffers used by
        # environments that are still in their previous episode.  Otherwise a
        # short replacement window can silently truncate their target cursor.
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
        non_cyclic_tasks = {
            str(value).upper() for value in getattr(self.cfg, "imitation_non_cyclic_tasks", ("BT05",))
        }
        for env_index, value in zip(ids.cpu().tolist(), values, strict=True):
            length = int(value["sequence_length"])
            for name in sequence_shapes:
                sequence = value[name].to(device=self.device, dtype=torch.float32)
                self._imitation_sequence_targets[name][env_index, :length] = sequence
                if length < max_length:
                    self._imitation_sequence_targets[name][env_index, length:] = sequence[-1]
            self._imitation_sequence_lengths[env_index] = length
            # The decoded sequence is shared per task to keep 4096-env reset
            # cost bounded.  Independent cyclic phase offsets prevent every
            # environment from receiving the exact same target frame at the
            # same control step.
            self._imitation_target_cursor[env_index] = int(
                torch.randint(max(1, length), (), device=self.device).item()
            )

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
            loop_enabled[env_index] = bool(
                getattr(self.cfg, "imitation_sequence_loop", True)
            ) and str(task_id).upper() not in non_cyclic_tasks
        # Keep the target sampler usable by lightweight test doubles and by
        # callers that only reuse its dataset buffering helpers.  The concrete
        # environment always provides this method; a missing optional hook
        # must not prevent reset targets from being installed.
        apply_speed_targets = getattr(self, "_apply_task_speed_targets", None)
        if callable(apply_speed_targets):
            apply_speed_targets(ids, task_ids)
        self._write_current_targets(ids)

    def _write_current_targets(self, env_ids: torch.Tensor | None = None) -> None:
        """Gather the current sequence frame into the manager reward cache."""

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
        """Advance non-terminal environments by one 50 Hz imitation frame."""

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
        """Step the manager environment and advance the recorded target time."""

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
        else:  # pragma: no cover - Isaac Lab always returns the five-tuple
            self._advance_target_sequences()
        return result

    def _emit_diagnostics_if_due(self) -> None:
        """Write one compact play-time snapshot at the configured interval."""

        if not bool(getattr(self.cfg, "diagnostics_enabled", False)):
            return
        interval = max(1, int(getattr(self.cfg, "diagnostics_interval", 20)))
        if self._diagnostics_step % interval:
            return
        try:
            reward = None
            try:
                # Recompute only at the sparse diagnostic interval so the
                # snapshot reports the six-channel imitation score rather
                # than the full locomotion reward sum.
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
        """Return the latest compact multimodal snapshot for external viewers."""

        reward = getattr(getattr(self, "reward_manager", None), "_reward_buf", None)
        return build_imitation_diagnostics(self, reward)

    def update_event_observation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a fixed event feature tensor for the current simulation step."""

        if not self.sensor_enabled("event") or self._dvs_runtime is None:
            zeros = torch.zeros((self.num_envs, EVENT_FEATURES), device=self.device)
            valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self._latest_event_hist = zeros
            self._latest_event_valid = valid
            return zeros, valid
        # Observation and reward managers can ask for this term in the same
        # simulation step.  Process EVIS once, otherwise the second query would
        # advance the asynchronous reference twice and double-count events.
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
                    # The first frame initializes the DVS reference and has no
                    # event mask.  Keep it invalid until a real frame arrives.
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
            # Each plane is already [N,H,W].  Stacking on the channel axis is
            # therefore stable for both one-env and cloned 4096-env scenes.
            stacked = torch.stack(planes, dim=1).to(device=self.device)
            if stacked.ndim != 4 or stacked.shape[0] != self.num_envs or stacked.shape[1] != 4:
                raise RuntimeError(f"EVIS masks produced invalid batch shape {tuple(stacked.shape)}")
            stacked = torch.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
            # [N, 4, H, W] -> [N, 4, 2, 4] -> 32 compact event features.
            pooled = torch.nn.functional.adaptive_avg_pool2d(stacked, (2, 4))
            value = torch.log1p(pooled).reshape(self.num_envs, EVENT_FEATURES)
            self._latest_event_hist = value
            self._latest_event_valid = torch.ones(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )
            self._event_last_process_time = current_time
        except Exception as exc:  # pragma: no cover - depends on optional EVIS version
            _LOGGER.debug("event runtime did not return a fixed histogram: %s", exc)
            self._event_last_process_time = current_time
            self._latest_event_valid.zero_()
        return self._latest_event_hist, self._latest_event_valid

    def close(self):
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


# Backwards-compatible import name for callers that expect a task module class.
MyExoskeletonImitationRLEnv = MyExoskeletonImitationEnv

__all__ = [
    "MyExoskeletonImitationEnv",
    "MyExoskeletonImitationRLEnv",
    "TASK_SPEED_M_S",
    "build_imitation_diagnostics",
]
