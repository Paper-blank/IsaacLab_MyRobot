# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the my_exoskeleton playback and training entrypoints."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "play_my_exoskeleton.py"
_SPEC = importlib.util.spec_from_file_location("play_my_exoskeleton", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_TRAIN_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "train_my_exoskeleton.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location("train_my_exoskeleton", _TRAIN_SCRIPT_PATH)
assert _TRAIN_SPEC is not None and _TRAIN_SPEC.loader is not None
_TRAIN_MODULE = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(_TRAIN_MODULE)


def test_show_imu_enables_all_sensor_debug_overrides_for_kit():
    """The convenience flag should stay scoped to one visible Kit environment."""
    argv = _MODULE._enable_imu_probes(["--num_envs=1", "--visualizer", "kit", "--show-imu"])

    assert "--show-imu" not in argv
    assert argv[-6:] == [f"env.scene.{sensor_name}.debug_vis=True" for sensor_name in _MODULE._IMU_SENSOR_NAMES]


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--num_envs=2", "--visualizer", "kit", "--show-imu"], "--num_envs=1"),
        (["--num_envs=1", "--visualizer", "none", "--show-imu"], "--visualizer kit"),
    ],
)
def test_show_imu_rejects_unsupported_playback_modes(argv, message):
    """Probe telemetry is meaningful only in the interactive one-env inspection flow."""
    with pytest.raises(ValueError, match=message):
        _MODULE._enable_imu_probes(argv)


def test_usd_bootstrap_reexecs_with_kit_runtime_paths(tmp_path, monkeypatch):
    """The Kit USD runtime should be selected before Isaac Lab imports are evaluated."""
    site_packages = tmp_path / "site-packages"
    usd_root = site_packages / "isaacsim" / "extscache" / "omni.usd.libs-test"
    (usd_root / "pxr").mkdir(parents=True)
    (usd_root / "bin").mkdir()
    monkeypatch.setattr(_MODULE.sysconfig, "get_paths", lambda: {"purelib": str(site_packages)})
    monkeypatch.delenv(_MODULE._KIT_USD_BOOTSTRAP_ENV, raising=False)
    monkeypatch.setenv("PYTHONPATH", "existing-python-path")
    monkeypatch.setenv("LD_LIBRARY_PATH", "existing-library-path")

    captured = {}

    def fake_execvpe(executable, argv, env):
        captured.update(executable=executable, argv=argv, env=env)

    monkeypatch.setattr(_MODULE.os, "execvpe", fake_execvpe)
    _MODULE._bootstrap_isaacsim_usd()

    assert captured["executable"] == _MODULE.sys.executable
    assert captured["env"][_MODULE._KIT_USD_BOOTSTRAP_ENV] == "1"
    assert captured["env"]["PYTHONPATH"].split(_MODULE.os.pathsep)[:2] == [
        str(usd_root),
        "existing-python-path",
    ]
    assert captured["env"]["LD_LIBRARY_PATH"].split(_MODULE.os.pathsep)[:2] == [
        str(usd_root / "bin"),
        "existing-library-path",
    ]


def test_usd_bootstrap_guard_avoids_reexec(monkeypatch):
    """A restarted process must not recursively restart itself."""
    monkeypatch.setenv(_MODULE._KIT_USD_BOOTSTRAP_ENV, "1")

    def fail_execvpe(*_args, **_kwargs):
        pytest.fail("USD bootstrap attempted to re-exec more than once")

    monkeypatch.setattr(_MODULE.os, "execvpe", fail_execvpe)
    _MODULE._bootstrap_isaacsim_usd()


@pytest.mark.parametrize("module", [_MODULE, _TRAIN_MODULE])
def test_convenience_task_and_dataset_flags_become_backend_safe_overrides(module):
    """Custom aliases must be consumed before RL-Games forwards arguments to Hydra."""
    argv = module._consume_exoskeleton_cli_args(
        [
            "--task_id=Isaac-MyExoskeleton-Flat-v0",
            "--imitation_dataset",
            "/tmp/reference walk.npz",
            "--max_iterations=2",
        ]
    )

    assert "--task_id=Isaac-MyExoskeleton-Flat-v0" not in argv
    assert "--imitation_dataset" not in argv
    task_index = argv.index("--task")
    assert argv[task_index + 1] == "Isaac-MyExoskeleton-Flat-v0"
    assert argv[-1] == "env.commands.base_velocity.reference_data_path='/tmp/reference walk.npz'"
    assert "--max_iterations=2" in argv


@pytest.mark.parametrize("module", [_MODULE, _TRAIN_MODULE])
def test_convenience_cli_rejects_conflicting_selectors(module):
    """Ambiguous task or dataset sources should fail before Kit starts."""
    with pytest.raises(ValueError, match="different tasks"):
        module._consume_exoskeleton_cli_args(["--task", "Isaac-MyExoskeleton-Flat-v0", "--task_id", "other-task"])
    with pytest.raises(ValueError, match="dataset selector conflicts"):
        module._consume_exoskeleton_cli_args(
            [
                "--dataset=/tmp/a.npz",
                "env.commands.base_velocity.reference_data_path=/tmp/b.npz",
            ]
        )


@pytest.mark.parametrize("module", [_MODULE, _TRAIN_MODULE])
def test_dataset_task_id_is_an_imitation_selector(module):
    """BT/AT IDs select recordings while --task remains the Gym environment."""
    argv = module._consume_exoskeleton_cli_args(
        [
            "--task",
            "Isaac-MyExoskeleton-Imitation-v0",
            "--task_id",
            "BT02",
            "--imitation_dataset",
            "/tmp/Data",
        ]
    )

    assert "--task_id" not in argv
    assert argv[:2] == ["--task", "Isaac-MyExoskeleton-Imitation-v0"]
    assert "env.imitation_task_id=BT02" in argv
    assert "env.commands.base_velocity.task_id=BT02" in argv
    assert "env.imitation_dataset_path='/tmp/Data'" in argv


@pytest.mark.parametrize("module", [_MODULE, _TRAIN_MODULE])
def test_sensor_switches_are_consumed_as_environment_overrides(module):
    """Both entrypoints accept equals and spaced boolean sensor switches."""

    argv = module._consume_sensor_cli_args(
        [
            "--imu=0",
            "--rgb",
            "false",
            "--event=1",
            "--contact",
            "true",
            "--num_envs=4096",
        ]
    )

    assert "--imu=0" not in argv
    assert "--rgb" not in argv
    assert argv[-5:] == [
        "env.use_imu=False",
        "env.use_rgb=False",
        "env.use_event=True",
        "env.use_contact=True",
        "--num_envs=4096",
    ]


@pytest.mark.parametrize("module", [_MODULE, _TRAIN_MODULE])
def test_sensor_switches_default_to_environment_configuration(module):
    """No switch means no override, preserving config defaults (all enabled)."""

    assert module._consume_sensor_cli_args(["--headless"]) == ["--headless"]


@pytest.mark.parametrize("module", [_MODULE, _TRAIN_MODULE])
def test_disabled_imitation_camera_switches_remove_camera_entities_before_launch(module):
    """Disabled RGB/Event channels must not trigger launcher camera rendering."""

    argv = [
        "--task",
        "Isaac-MyExoskeleton-Imitation-v0",
        "env.use_rgb=False",
        "env.use_event=False",
        "env.scene.left_rgb_camera=custom",
    ]

    result = module._apply_sensor_scene_overrides(argv)

    assert "env.scene.left_rgb_camera=custom" not in result
    assert result.count("env.scene.left_rgb_camera=null") == 1
    assert result.count("env.scene.right_rgb_camera=null") == 1
    assert result.count("env.scene.left_event_camera=null") == 1
    assert result.count("env.scene.right_event_camera=null") == 1


@pytest.mark.parametrize("module", [_MODULE, _TRAIN_MODULE])
def test_disabled_camera_switches_do_not_mutate_legacy_flat_task(module):
    """The flat task has no camera entities and keeps its normal Hydra args."""

    argv = ["--task", "Isaac-MyExoskeleton-Flat-v0", "env.use_rgb=False"]
    assert module._apply_sensor_scene_overrides(argv) == argv


@pytest.mark.parametrize("module", [_MODULE, _TRAIN_MODULE])
@pytest.mark.parametrize("argument", ["--imu", "--rgb", "--event", "--contact"])
def test_sensor_switch_requires_value(module, argument):
    with pytest.raises(ValueError, match="boolean value"):
        module._consume_sensor_cli_args([argument, "--headless"])


@pytest.mark.parametrize("module", [_MODULE, _TRAIN_MODULE])
def test_sensor_switch_rejects_invalid_and_conflicting_values(module):
    with pytest.raises(ValueError, match="expects one of"):
        module._consume_sensor_cli_args(["--imu=maybe"])
    with pytest.raises(ValueError, match="conflicting --imu values"):
        module._consume_sensor_cli_args(["--imu=0", "--imu", "1"])


def test_diagnostics_flags_become_imitation_environment_overrides():
    """Playback telemetry flags stay out of the RL-Games parser."""

    argv = _MODULE._consume_diagnostics_args(
        [
            "--task",
            "Isaac-MyExoskeleton-Imitation-v0",
            "--diagnostics",
            "--diagnostics_interval=7",
            "--diagnostics-output",
            "/tmp/imitation diagnostics.jsonl",
            "--visualizer",
            "viser",
        ]
    )

    assert "--diagnostics" not in argv
    assert "--diagnostics_interval=7" not in argv
    assert "--diagnostics-output" not in argv
    assert "env.diagnostics_enabled=True" in argv
    assert "env.diagnostics_interval=7" in argv
    assert "env.diagnostics_output_path='/tmp/imitation diagnostics.jsonl'" in argv
    assert argv[-2:] == ["env.diagnostics_interval=7", "env.diagnostics_output_path='/tmp/imitation diagnostics.jsonl'"]


def test_diagnostics_interval_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive integer"):
        _MODULE._consume_diagnostics_args(["--diagnostics-interval", "0"])


def test_ruleresult_flags_become_environment_overrides():
    """Playback can request the structured 24-value joint result without Hydra seeing custom flags."""

    argv = _MODULE._consume_ruleresult_args(
        [
            "--ruleresult",
            "--ruleresult-interval=3",
            "--ruleresult-env-id",
            "2",
            "--ruleresult-output",
            "/tmp/ruleresult.jsonl",
            "--visualizer",
            "none",
        ]
    )

    assert "--ruleresult" not in argv
    assert "env.ruleresult_enabled=True" in argv
    assert "env.ruleresult_interval=3" in argv
    assert "env.ruleresult_env_id=2" in argv
    assert "env.ruleresult_output_path='/tmp/ruleresult.jsonl'" in argv
    assert argv[-4:] == [
        "env.ruleresult_enabled=True",
        "env.ruleresult_interval=3",
        "env.ruleresult_env_id=2",
        "env.ruleresult_output_path='/tmp/ruleresult.jsonl'",
    ]


def test_ruleresult_relative_output_is_anchored_to_launch_directory(tmp_path, monkeypatch):
    """A relative output path must not be rebased under the checkpoint log directory."""
    monkeypatch.chdir(tmp_path)

    argv = _MODULE._consume_ruleresult_args(["--ruleresult-output", "logs/ruleresult.jsonl"])

    assert argv[-2:] == [
        "env.ruleresult_enabled=True",
        f"env.ruleresult_output_path='{(tmp_path / 'logs' / 'ruleresult.jsonl').resolve()}'",
    ]


def test_ruleresult_protocol_flags_become_environment_overrides(tmp_path, monkeypatch):
    """Binary output options must be consumed before the RL-Games parser sees them."""
    monkeypatch.chdir(tmp_path)

    argv = _MODULE._consume_ruleresult_args(
        [
            "--ruleresult-format",
            "protocol_55aa",
            "--ruleresult-protocol-config",
            "protocol.json",
            "--ruleresult-output",
            "frames.bin",
        ]
    )

    assert argv == [
        "env.ruleresult_enabled=True",
        f"env.ruleresult_output_path='{(tmp_path / 'frames.bin').resolve()}'",
        "env.ruleresult_output_format=protocol_55aa",
        f"env.ruleresult_protocol_config_path='{(tmp_path / 'protocol.json').resolve()}'",
    ]


def test_ruleresult_interval_and_env_id_validate_values():
    with pytest.raises(ValueError, match="positive integer"):
        _MODULE._consume_ruleresult_args(["--ruleresult-interval", "0"])
    with pytest.raises(ValueError, match="non-negative integer"):
        _MODULE._consume_ruleresult_args(["--ruleresult-env-id", "-1"])


def test_ruleresult_timing_and_prediction_flags_become_overrides():
    argv = _MODULE._consume_ruleresult_args(
        [
            "--ruleresult-use-actual-dt",
            "--ruleresult-prediction-mode",
            "kinematic",
            "--ruleresult-prediction-horizon",
            "4",
            "--ruleresult-prediction-dt=0.02",
        ]
    )

    assert "env.ruleresult_enabled=True" in argv
    assert "env.ruleresult_time_mode=wall_clock" in argv
    assert "env.ruleresult_prediction_mode=kinematic" in argv
    assert "env.ruleresult_prediction_horizon=4" in argv
    assert "env.ruleresult_prediction_dt_s=0.02" in argv


def test_ruleresult_prediction_flags_validate_values():
    with pytest.raises(ValueError, match="non-negative integer"):
        _MODULE._consume_ruleresult_args(["--ruleresult-prediction-horizon", "-1"])
    with pytest.raises(ValueError, match="positive number"):
        _MODULE._consume_ruleresult_args(["--ruleresult-prediction-dt", "0"])
    with pytest.raises(ValueError, match="none, kinematic, or supplied"):
        _MODULE._consume_ruleresult_args(["--ruleresult-prediction-mode", "actor"])


def test_train_entrypoint_bootstraps_kit_usd_before_imports(tmp_path, monkeypatch):
    """Training must select the same Kit USD runtime as playback."""
    site_packages = tmp_path / "site-packages"
    usd_root = site_packages / "isaacsim" / "extscache" / "omni.usd.libs-test"
    (usd_root / "pxr").mkdir(parents=True)
    (usd_root / "bin").mkdir()
    monkeypatch.setattr(_TRAIN_MODULE.sysconfig, "get_paths", lambda: {"purelib": str(site_packages)})
    monkeypatch.delenv(_TRAIN_MODULE._KIT_USD_BOOTSTRAP_ENV, raising=False)

    captured = {}

    def fake_execvpe(executable, argv, env):
        captured.update(executable=executable, argv=argv, env=env)

    monkeypatch.setattr(_TRAIN_MODULE.os, "execvpe", fake_execvpe)
    _TRAIN_MODULE._bootstrap_isaacsim_usd()

    assert captured["executable"] == _TRAIN_MODULE.sys.executable
    assert captured["env"][_TRAIN_MODULE._KIT_USD_BOOTSTRAP_ENV] == "1"
    assert captured["env"]["PYTHONPATH"].split(_TRAIN_MODULE.os.pathsep)[0] == str(usd_root)
    assert captured["env"]["LD_LIBRARY_PATH"].split(_TRAIN_MODULE.os.pathsep)[0] == str(usd_root / "bin")
