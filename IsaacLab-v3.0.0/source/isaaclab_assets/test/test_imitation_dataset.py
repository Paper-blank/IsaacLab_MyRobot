"""Tests for the dependency-light vision/IMU imitation dataset loader."""

from __future__ import annotations

import csv
import json

import numpy as np
import torch

from isaaclab_assets.tasks.imitation_dataset import (
    DatasetFormatError,
    ExoskeletonImitationDataset,
    SensorCoordinateCalibration,
    load_direction_calibration,
    load_imu_mapping,
    load_task_mapping,
    resolve_imu_mapping,
)


def _write_imu(path, sensor_id: int, count: int = 6):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "timestamp_decoded",
                "acc_x_mps2",
                "acc_y_mps2",
                "acc_z_mps2",
                "gyro_x_dps",
                "gyro_y_dps",
                "gyro_z_dps",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
            ]
        )
        for index in range(count):
            writer.writerow(
                [
                    f"device+00T00:00:00.{index * 20:03d}",
                    0,
                    0,
                    9.81,
                    sensor_id,
                    0,
                    0,
                    index,
                    0,
                    0,
                ]
            )


def _write_trial(root):
    trial = root / "records" / "P001" / "2026_01_01" / "BT02" / "Trial01"
    trial.mkdir(parents=True)
    for sensor_id in range(1, 7):
        _write_imu(trial / f"P001_BT02_imu_{sensor_id:02d}.csv", sensor_id)
    np.savez(
        trial / "motion.npz",
        t_s=np.arange(6, dtype=np.float32) * 0.02,
        joint_pos_rad=np.zeros((6, 8), dtype=np.float32),
    )
    np.save(trial / "left_rgb.npy", np.full((6, 8, 8, 3), 64, dtype=np.uint8))
    np.savez(
        trial / "left_event.npz",
        events=np.asarray(
            [[1, 2, 0.005, 1], [3, 4, 0.025, -1], [319, 239, 0.035, 1]], dtype=np.float32
        ),
    )
    (trial / "P001_BT02.json").write_text(
        json.dumps(
            {
                "task": "BT02",
                "subject": {"uid": "P001"},
                "files": {
                    **{
                        f"imu_{sensor_id:02d}_csv": f"P001_BT02_imu_{sensor_id:02d}.csv"
                        for sensor_id in range(1, 7)
                    },
                    "left_rgb": "left_rgb.npy",
                    "left_event": "left_event.npz",
                },
            }
        ),
        encoding="utf-8",
    )


def test_mapping_files_and_usd_alias_resolution():
    assert set(load_task_mapping()) == {"BT01", "BT02", "BT03", "BT04", "BT05", "AT01", "AT02", "AT03", "AT04", "AT05"}
    mapping = load_imu_mapping()
    assert [mapping[f"imu_{index:02d}"]["sensor_id"] for index in range(1, 7)] == [1, 2, 3, 4, 5, 6]
    assert [mapping[f"imu_{index:02d}"]["robot_link"] for index in range(1, 7)] == [
        "left_foot", "right_foot", "left_shank", "right_shank", "left_thigh", "right_thigh"
    ]
    assert mapping["imu_01"]["robot_link"] == "left_foot"
    assert resolve_imu_mapping(["Lfoot", "Rfoot", "Lleg", "Rleg", "Lthigh", "Rthigh"]) == {
        "imu_01": "Lfoot",
        "imu_02": "Rfoot",
        "imu_03": "Lleg",
        "imu_04": "Rleg",
        "imu_05": "Lthigh",
        "imu_06": "Rthigh",
    }


def test_default_direction_calibration_flips_sagittal_imu_and_hip_knee_targets(tmp_path):
    """Recorded forward/sagittal signs are converted before PPO targets are built."""

    _write_trial(tmp_path)
    trial = tmp_path / "records" / "P001" / "2026_01_01" / "BT02" / "Trial01"
    _write_imu(trial / "P001_BT02_imu_01.csv", sensor_id=1)
    # Replace the first IMU with non-zero values in every calibrated channel.
    with (trial / "P001_BT02_imu_01.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "timestamp_decoded",
                "acc_x_mps2",
                "acc_y_mps2",
                "acc_z_mps2",
                "gyro_x_dps",
                "gyro_y_dps",
                "gyro_z_dps",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
            ]
        )
        for index in range(6):
            writer.writerow(
                [
                    f"device+00T00:00:00.{index * 20:03d}",
                    1.0,
                    2.0,
                    3.0,
                    10.0,
                    20.0,
                    30.0,
                    10.0,
                    20.0,
                    30.0,
                ]
            )
    q = np.tile(np.arange(8, dtype=np.float32), (6, 1))
    np.savez(
        trial / "motion.npz",
        t_s=np.arange(6, dtype=np.float32) * 0.02,
        joint_pos_rad=q,
        joint_vel_rad_s=np.ones_like(q),
    )

    calibration = load_direction_calibration()
    assert calibration.joint_sign == (1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0)
    dataset = ExoskeletonImitationDataset(
        tmp_path,
        task_id="BT02",
        window_size=1,
        random_window=False,
        load_rgb=False,
        load_event=False,
    )
    sample = dataset[0]

    expected_joint = torch.tensor([0.0, 1.0, -2.0, 3.0, 4.0, -5.0, 6.0, 7.0])
    expected_velocity = torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
    assert torch.allclose(sample["target_motion"][0], expected_joint)
    assert torch.allclose(sample["target_joint_velocity"][0], expected_velocity)

    degrees_to_rad = np.pi / 180.0
    expected_imu = torch.tensor(
        [-20.0, 10.0, -30.0, -20.0, 10.0, -30.0, -2.0, 1.0, -3.0], dtype=torch.float32
    )
    expected_imu[:6] = expected_imu[:6] * degrees_to_rad
    assert torch.allclose(sample["target_imu"][0, 0], expected_imu, atol=1.0e-6)
    dataset.close()


def test_leg_imu_axis_mapping_matches_dataset_to_joint_rules():
    """Every odd/even recorder ID selects the left/right leg matrix."""

    calibration = load_direction_calibration()
    values = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)

    expected_right = np.asarray([[2.0, 1.0, -3.0]])
    expected_left = np.asarray([[-2.0, 1.0, -3.0]])
    for sensor_index in (1, 3, 5):
        left = calibration.apply_imu(values, values, values, sensor_index=sensor_index)
        assert all(np.array_equal(channel, expected_left) for channel in left)
    for sensor_index in (2, 4, 6):
        right = calibration.apply_imu(values, values, values, sensor_index=sensor_index)
        assert all(np.array_equal(channel, expected_right) for channel in right)
    assert calibration.imu_sensor_sides == ("left", "right", "left", "right", "left", "right")


def test_imu_mounting_chain_composes_before_existing_axis_mapping():
    """R_joint_link @ R_link_sensor is explicit and identity by default."""

    calibration = load_direction_calibration()
    values = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    right_matrix = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    np.testing.assert_allclose(calibration.imu_transform_matrix(sensor_index=2), right_matrix)
    mapped = calibration.apply_imu(values, values, values, sensor_index=2)
    np.testing.assert_allclose(mapped[1], values @ right_matrix.T)

    # A measured mounting rotation is applied on the left of R_sensor_IMU.
    quarter_turn = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    calibrated = SensorCoordinateCalibration(
        imu_joint_link_rotations={f"imu_{index:02d}": quarter_turn for index in range(1, 7)},
        imu_link_sensor_rotations={f"imu_{index:02d}": identity for index in range(1, 7)},
    )
    expected = np.asarray(quarter_turn) @ right_matrix
    np.testing.assert_allclose(calibrated.imu_transform_matrix(sensor_index=2), expected)
    np.testing.assert_allclose(calibrated.apply_imu(values, values, values, sensor_index=2)[1], values @ expected.T)


def test_rotation_matrix_orientation_mode_rejects_reflection():
    """The prescribed left component map is not a physical quaternion rotation."""

    calibration = SensorCoordinateCalibration(imu_orientation_mode="rotation_matrix")
    values = np.zeros((1, 3), dtype=np.float64)
    with np.testing.assert_raises(DatasetFormatError):
        calibration.apply_imu(values, values, values, sensor_index=1)


def test_dataset_applies_leg_axis_mapping_during_csv_loading(tmp_path):
    """The trial loader passes each recorder ID to the axis calibration."""

    _write_trial(tmp_path)
    trial = tmp_path / "records" / "P001" / "2026_01_01" / "BT02" / "Trial01"

    def write_vector(path, orientation, gyro, accel):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "timestamp_decoded",
                    "acc_x_mps2",
                    "acc_y_mps2",
                    "acc_z_mps2",
                    "gyro_x_dps",
                    "gyro_y_dps",
                    "gyro_z_dps",
                    "roll_deg",
                    "pitch_deg",
                    "yaw_deg",
                ]
            )
            for index in range(6):
                writer.writerow(
                    [
                        f"device+00T00:00:00.{index * 20:03d}",
                        *accel,
                        *gyro,
                        *orientation,
                    ]
                )

    for sensor_id in range(1, 7):
        write_vector(
            trial / f"P001_BT02_imu_{sensor_id:02d}.csv",
            (1.0, 2.0, 3.0),
            (10.0, 20.0, 30.0),
            (4.0, 5.0, 6.0),
        )

    dataset = ExoskeletonImitationDataset(
        tmp_path,
        task_id="BT02",
        window_size=1,
        random_window=False,
        load_rgb=False,
        load_event=False,
    )
    target = dataset[0]["target_imu"][0].numpy()
    degrees_to_rad = np.pi / 180.0
    expected_left = np.concatenate(
        (
            np.asarray([-2.0, 1.0, -3.0]) * degrees_to_rad,
            np.asarray([-20.0, 10.0, -30.0]) * degrees_to_rad,
            np.asarray([-5.0, 4.0, -6.0]),
        )
    )
    expected_right = np.concatenate(
        (
            np.asarray([2.0, 1.0, -3.0]) * degrees_to_rad,
            np.asarray([20.0, 10.0, -30.0]) * degrees_to_rad,
            np.asarray([5.0, 4.0, -6.0]),
        )
    )
    for sensor_index in (0, 2, 4):
        np.testing.assert_allclose(target[sensor_index], expected_left)
    for sensor_index in (1, 3, 5):
        np.testing.assert_allclose(target[sensor_index], expected_right)
    dataset.close()


def test_dataset_ignores_imu_07_to_11(tmp_path):
    """Extra recorder channels never enter the six-channel target contract."""

    _write_trial(tmp_path)
    trial = tmp_path / "records" / "P001" / "2026_01_01" / "BT02" / "Trial01"
    for sensor_id in range(7, 12):
        (trial / f"P001_BT02_imu_{sensor_id:02d}.csv").write_text("not,a,valid,imu,file\n", encoding="utf-8")

    dataset = ExoskeletonImitationDataset(
        tmp_path,
        task_id="BT02",
        window_size=1,
        random_window=False,
        load_rgb=False,
        load_event=False,
    )
    sample = dataset[0]
    assert sample["target_imu"].shape[1:] == (6, 9)
    dataset.close()


def test_dataset_synchronizes_imu_motion_rgb_and_events(tmp_path):
    _write_trial(tmp_path)
    dataset = ExoskeletonImitationDataset(
        tmp_path,
        task_id="BT02",
        window_size=2,
        random_window=False,
        event_size=(8, 8),
        load_rgb=True,
        load_event=True,
    )
    sample = dataset[0]
    assert sample["target_imu"].shape == (2, 6, 9)
    assert sample["target_imu_gyro"].shape == (2, 6, 3)
    assert sample["target_motion"].shape == (2, 8)
    assert bool(sample["motion_valid"])
    assert bool(sample["rgb_valid"][0])
    assert bool(sample["event_valid"])
    assert sample["target_rgb"].shape == (2, 2, 3, 256, 256)
    assert sample["target_event"].shape == (2, 2, 2, 8, 8)
    assert sample["target_event_stream"].shape == (3, 4)
    assert sample["target_event_frame_index"].shape == (3,)
    assert sample["target_event_camera_index"].shape == (3,)
    assert sample["target_event"][1, 0, 1, 7, 7] == 1.0
    assert torch.isfinite(sample["target_imu"]).all()
    dataset.close()


def test_quiet_event_window_is_valid_when_event_file_decodes(tmp_path):
    """A quiet DVS interval is distinct from an unavailable event decoder."""

    _write_trial(tmp_path)
    dataset = ExoskeletonImitationDataset(
        tmp_path,
        task_id="BT02",
        window_size=1,
        random_window=False,
        event_size=(8, 8),
        load_rgb=False,
        load_event=True,
    )
    sample = dataset.sample_at("BT02", timestamp_s=0.10, window_size=1)
    assert bool(sample["event_valid"])
    assert sample["target_event_stream"].shape == (0, 4)
    assert float(sample["target_event"].sum()) == 0.0
    dataset.close()
