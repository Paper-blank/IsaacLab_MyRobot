# 外骨骼模仿数据读取修改指南

本文说明当前项目的数据读取链路，以及在增加传感器字段、修改坐标方向、接入新的动作文件或排查数据缺失时应修改的位置。说明以 Isaac Lab 3.0 项目中的
`ExoskeletonImitationDataset` 为准，不依赖 Isaac Sim，所有检查命令都可以在 Kit 启动前执行。

项目根目录：

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
```

## 1. 数据读取流程

当前读取器的顺序是：

```text
数据集根目录
  -> 递归发现 Trial* 目录
  -> 读取 metadata JSON 和 task_mapping.yaml
  -> 读取 IMU-01..IMU-06 CSV
  -> 对每一路 IMU 做字段解析、单位转换、方向校准
  -> 求六路 IMU 的共同时间区间
  -> 按 target_dt_s 重采样，默认 0.02 s（50 Hz）
  -> 读取压力、关节运动、RGB、Event 可选通道
  -> 生成 TrialRecord
  -> sample() 取随机或时间定位窗口
  -> 环境 reset 时把窗口写入 imitation targets
```

主要实现文件：

| 文件 | 作用 |
| --- | --- |
| `source/isaaclab_assets/tasks/imitation_dataset.py` | 发现 trial、解析和同步数据、窗口采样 |
| `task_mapping.yaml` | 十个 BT/AT 任务的编号和名称 |
| `sensor_coordinate_mapping.yaml` | IMU 与关节方向符号 |
| `imu_mapping.yaml` | IMU 编号到机器人 link 的映射 |
| `source/isaaclab_assets/tasks/my_exoskeleton_imitation.py` | 将 Dataset 样本写入环境目标和奖励 |
| `source/isaaclab_assets/envs/my_exoskeleton_imitation_env_cfg.py` | Dataset 路径、任务和窗口的 Hydra 配置 |

不要在 `my_exoskeleton_imitation.py` 中重新解析 CSV。数据格式和校准应集中在 Dataset 层，否则训练和离线检查会得到不同的目标。

## 2. 目录和任务识别

推荐目录结构：

```text
/home/lqf/MyCode/Data/
└── records/<subject>/<date>/<task_id>/TrialXX/
    ├── metadata.json
    ├── *_imu_01.csv
    ├── *_imu_02.csv
    ├── ...
    ├── *_imu_06.csv
    ├── left_*recording*.mp4       # 可选 RGB
    ├── right_*recording*.mp4      # 可选 RGB
    ├── left_*.aedat4              # 可选 Event
    ├── right_*.aedat4             # 可选 Event
    └── motion.csv/.npz/.pt        # 可选 [T, 8] 关节轨迹
```

读取器只递归查找名字匹配 `Trial*` 的目录。任务 ID 的优先级是：

1. trial 内 metadata JSON 的 `task` 字段；
2. trial 路径中最后一个完全匹配的 `BT01`...`AT05` 目录名。

metadata 中可以显式指定文件：

```json
{
  "task": "BT02",
  "subject": {"uid": "P01"},
  "files": {
    "imu_01_csv": "imu_01.csv",
    "imu_02_csv": "imu_02.csv",
    "left_rgb": "left_daheng_recording.mp4",
    "right_rgb": "right_daheng_recording.mp4",
    "left_event": "left_dvxplorer_recording.aedat4",
    "right_event": "right_dvxplorer_recording.aedat4",
    "pressure_left_csv": "left_pressure.csv",
    "pressure_right_csv": "right_pressure.csv"
  }
}
```

没有 metadata 文件时，读取器会按文件名尝试：`*_imu_01.csv` 到 `*_imu_06.csv`、`left_*recording*.mp4`、`right_*recording*.mp4`，以及相应的 event 文件。

## 3. 创建 Dataset

最小的 IMU 检查：

```bash
uv run python - <<'PY'
from isaaclab_assets.tasks.imitation_dataset import ExoskeletonImitationDataset

dataset = ExoskeletonImitationDataset(
    "/home/lqf/MyCode/Data",
    task_id="BT02",
    window_size=32,
    target_dt_s=0.02,
    random_window=True,
    random_task_sampling=False,
    load_rgb=False,
    load_event=False,
)
print("trials:", len(dataset.trials))
print("tasks:", dataset.task_ids)
sample = dataset.sample(task_id="BT02", window_size=4)
print("target_imu:", tuple(sample["target_imu"].shape))
print("imu_valid:", sample["imu_valid"].tolist())
print("motion_valid:", bool(sample["motion_valid"]))
dataset.close()
PY
```

构造函数的关键参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `root` | 必填 | 数据集根目录 |
| `task_id` | `None` | 一个任务、任务列表或全部任务 |
| `window_size` | `1` | 每次取出的 policy-rate 帧数 |
| `target_dt_s` | `0.02` | 同步时间步，50 Hz |
| `random_window` | `True` | 是否随机选窗口起点 |
| `random_task_sampling` | `False` | `task_id=all` 时是否先均匀抽任务 |
| `steps_per_epoch` | `None` | 覆盖 Dataset 长度 |
| `seed` | `0` | 可复现随机采样种子 |
| `rgb_size` | `(256, 256)` | 输出 RGB 的 `(height, width)` |
| `event_size` | `(64, 64)` | Event histogram 的 `(height, width)` |
| `event_source_size` | `(240, 320)` | 原始事件传感器 `(height, width)` |
| `load_rgb` | `True` | 是否在 sample 时解码 RGB |
| `load_event` | `True` | 是否解码 AEDAT4/事件文件 |
| `strict` | `False` | trial 出错时是否立即抛异常 |
| `task_mapping_path` | 自动 | 自定义任务映射 YAML |
| `calibration_path` | 自动 | 自定义方向校准 YAML |

训练环境通常使用：

```text
env.imitation_dataset_path=/home/lqf/MyCode/Data
env.imitation_task_id=BT02
env.imitation_window_size=32
env.imitation_calibration_path=/absolute/path/sensor_coordinate_mapping.yaml
```

## 4. IMU CSV 格式和单位

每个 IMU CSV 至少需要一列时间和以下三组通道。字段名可以使用下列别名：

```text
时间: timestamp_decoded | timestamp | t_s | time_s | time
加速度: acc_x_mps2/acc_x/ax, acc_y_mps2/acc_y/ay, acc_z_mps2/acc_z/az
角速度: gyro_x_rad_s/gyro_x_rads/gyro_x/gx, ... y ..., ... z ...
姿态: roll_rad/roll_rads/roll_deg/roll,
      pitch_rad/pitch_rads/pitch_deg/pitch,
      yaw_rad/yaw_rads/yaw_deg/yaw
```

解析规则：

- 没有时间列时按行号乘 `target_dt_s` 生成时间。
- ISO/录制器时间字符串会被转换为秒；`us`/`µs` 后缀会转换为秒。
- 角速度只有在列名明确包含 `rad` 时按 rad/s 处理，否则按 deg/s 转为 rad/s。
- 姿态只有在列名明确包含 `rad` 时按 rad 处理，否则按 deg 转为 rad。
- 缺失单元格会对有限值插值；整列缺失则填零。
- 相同时间戳会去重，保留第一次出现的样本。
- 字段匹配要求下划线或 `[` 边界，避免 `gy` 被错误匹配到 `gyro_x`。

最终每帧六路 IMU 的排列为：

```text
[roll, pitch, yaw, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
```

输出 `target_imu` 的形状是 `[window, 6, 9]`，`imu_valid` 的形状是 `[6]`。

## 5. 修改方向校准

当前读取层默认加载项目根目录 `sensor_coordinate_mapping.yaml`：

```yaml
version: 1
coordinate_calibration:
  joint_sign: [1, 1, -1, 1, 1, -1, 1, 1]
  imu_sensor_sides: [left, right, left, right, left, right]
  # output_joint = matrix @ dataset_vector
  imu_axis_transforms:
    right: [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    left: [[0, -1, 0], [1, 0, 0], [0, 0, -1]]
  # R_exo = R_joint_link @ R_link_sensor @ R_sensor_IMU
  # 未提供实体安装标定时，两个安装旋转均为单位阵。
  R_joint_link:
    default: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  R_link_sensor:
    default: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  orientation_mode: euler_components
  imu_orientation_sign: [1, -1, 1]
  imu_gyro_sign: [-1, -1, 1]
  imu_accel_sign: [-1, -1, 1]
```

六路 IMU 的数据集编号与腿侧固定对应为：`IMU-01/03/05=left`、`IMU-02/04/06=right`。
具体位置为 `01/02=左右踝`、`03/04=左右膝`、`05/06=左右髋`。对腿部 IMU，读取器对姿态、角速度和加速度的三个轴都使用同一个
“数据集坐标 -> joint 坐标”矩阵：

| 传感器侧 | joint x | joint y | joint z |
| --- | --- | --- | --- |
| 右腿（IMU-02/04/06） | `dataset_y` | `dataset_x` | `-dataset_z` |
| 左腿（IMU-01/03/05） | `-dataset_y` | `dataset_x` | `-dataset_z` |

因此，输入向量 `[x_d, y_d, z_d]` 的输出分别为右腿 `[y_d, x_d, -z_d]`、左腿
`[-y_d, x_d, -z_d]`。当前六路均为腿部传感器，不再将 IMU-01 当作 base。矩阵在
`_read_imu_csv(..., sensor_index=index)` 中、插值和多路时间同步之前执行；不要再在奖励函数
或控制器中重复变换，否则会发生双重翻转。
配置中的 `imu_orientation_sign`、`imu_gyro_sign` 和 `imu_accel_sign` 仅兼容不传
`sensor_index` 的旧外部调用；数据集主流程不会将这些符号与轴矩阵叠加。

### 安装姿态链和姿态变换

读取器对角速度和线加速度使用完整链：

```text
R_exo = R_joint_link @ R_link_sensor @ R_sensor_IMU
omega_exo = R_exo @ omega_imu
a_exo = R_exo @ a_imu
```

`R_joint_link` 和 `R_link_sensor` 可以按 `imu_01`--`imu_06` 配置，必须是右手正交旋转矩阵（`det=+1`）。当前给出的 CSV、URDF 和 USD 资料没有包含六个实体传感器相对于 link 的安装姿态，因此默认是单位阵；这不是从数据推断出的测量结果，现场标定后应覆盖这两个字段。现有
`imu_axis_transforms` 保留用户指定的左右腿轴约定，其中左腿矩阵是一个反射（`det=-1`），所以默认
`orientation_mode: euler_components` 继续按已发布的欧拉分量约定变换姿态，保证历史数据兼容。只有当有效链为右手旋转时，才可切换为
`orientation_mode: rotation_matrix`，此模式按 `R_exo @ R_imu` 转换 XYZ 欧拉姿态；左腿反射矩阵在该模式下会被明确拒绝，避免产生无效四元数。

八个关节的固定顺序必须是：

```text
0 base_to_Rturn
1 Rturn_to_Rthigh
2 Rthigh_to_Rleg
3 Rleg_to_Rfoot
4 base_to_Lturn
5 Lturn_to_Lthigh
6 Lthigh_to_Lleg
7 Lleg_to_Lfoot
```

如果某个现场传感器安装方向不同，优先复制 YAML 并修改对应的 `imu_sensor_sides` 或
`imu_axis_transforms`，不要直接改 Python 常量：

```bash
cp sensor_coordinate_mapping.yaml /absolute/path/site_mapping.yaml
# 编辑 site_mapping.yaml 后运行：
uv run python - <<'PY'
from isaaclab_assets.tasks.imitation_dataset import ExoskeletonImitationDataset

dataset = ExoskeletonImitationDataset(
    "/home/lqf/MyCode/Data",
    task_id="BT02",
    window_size=2,
    load_rgb=False,
    load_event=False,
    calibration_path="/absolute/path/site_mapping.yaml",
)
sample = dataset.sample(task_id="BT02", random_window=False, window_size=2)
print(sample["target_imu"][0, :, :].numpy())
dataset.close()
PY
```

修改矩阵时保持“行是 joint 输出轴、列是数据集输入轴”的约定。每个矩阵必须是只含
`0/+1/-1` 的 3×3 signed-permutation matrix；加载器会拒绝缩放、剪切或重复轴。

校准只应在读取层执行一次。不要在 `imitation_reward.py`、环境 observation term 或动作控制器中再次对同一参考数据乘符号，否则会发生双重翻转。当前实现会在以下时机应用校准：

- `_read_imu_csv()`：CSV 解析后、插值和同步前；
- `_read_motion()`：关节位置和关节速度读取后；
- `_load_trial()`：把已校准的通道重采样到共同时间轴。

## 6. 关节运动文件

关节文件是可选的。没有有效文件时，sample 仍会返回零张量，但 `motion_valid=False`，关节模仿奖励应被屏蔽。

推荐使用 NPZ：

```python
import numpy as np

np.savez(
    "motion.npz",
    t_s=np.arange(T, dtype=np.float32) * 0.02,
    joint_pos_rad=np.zeros((T, 8), dtype=np.float32),
    # 可选
    joint_vel_rad_s=np.zeros((T, 8), dtype=np.float32),
)
```

也支持 JSON/PT/PTH 和 CSV。CSV 至少要有八列 `joint_pos_rad[0]` 到 `joint_pos_rad[7]`，或 `q_rad_0`、`q0`、`joint_0` 等兼容列名。

要求：

- 位置形状必须是 `[T, 8]`，单位 rad；如果输入是 `[8, T]`，读取器会转置。
- 时间列可省略，省略时按 `target_dt_s` 生成。
- 位置和速度会使用同一 `joint_sign` 校准。
- 没有速度时用位置对时间求梯度生成；时间至少需要两个有限样本。
- 文件名应包含 `motion`、`joint`、`kinematic`、`q_rad` 或 `qpos` 之一，才能被 `_discover_motion()` 自动发现。

如果使用了不含这些关键词的文件名，在 metadata 中增加显式 motion 字段前，最简单的做法是重命名为 `motion.npz`；不要在 `_load_trial()` 中写只针对某一条 trial 的硬编码路径。

## 7. RGB 和 Event 读取

### RGB

`load_rgb=True` 时，Dataset 在 sample 时按时间读取左右视频帧，并输出：

```text
target_rgb: [window, 2, 3, 256, 256]
rgb_valid: [2]  # 左、右相机是否成功读取
```

支持 MP4 等视频（优先 imageio，失败时尝试 OpenCV），也支持 NPY/NPZ 帧数组。数组可以使用 `frames`、`rgb` 或 `images` 作为 NPZ 键。像素会转换为 RGB、缩放到 `rgb_size` 并归一化到 `[0, 1]`。

### Event

Event 文件必须是真实事件流，输出每个事件的四列：

```text
(x, y, t_relative_s, polarity)
```

支持 `.aedat4`（需要 `dv-processing`）、`.npz`、`.npy`、`.csv` 和 `.json`。输出为：

```text
target_event: [window, 2, 2, H, W]
target_event_stream: [N, 4]
target_event_frame_index: [N]
target_event_camera_index: [N]
event_valid: scalar bool
```

读取器明确拒绝把 event preview MP4 当成事件流，也不会通过 RGB 帧差伪造 Event。事件时间戳的绝对值大于 `1e5` 时按微秒转换为秒，再减去该文件的最小时间。

## 8. 时间同步和窗口采样

六路 IMU 的共同时间区间为：

```text
start = max(每路起始时间)
end   = min(每路结束时间)
```

如果区间无交集，读取器会退回到最长一路的时间范围。随后以 `target_dt_s` 生成共同时间轴，并对姿态、角速度、加速度、压力和关节通道逐列线性插值。

常用采样接口：

```python
sample = dataset.sample(
    task_id="BT05",
    timestamp_s=3.0,
    window_size=32,
    random_window=False,
)

# 固定任务和时间的快捷方式
sample = dataset.sample_at("BT05", timestamp_s=3.0, window_size=32)
```

`random_task_sampling=True` 且 `task_id=None/all` 时，先在有数据的任务之间抽任务，再在该任务的 trial 中抽样。 `set_epoch(epoch)` 会改变可复现的随机序列，适合训练循环中每个 epoch 调用一次。

## 9. 修改读取器的正确方式

### 增加一个 IMU 字段

按下面顺序修改：

1. 在 `_read_imu_csv()` 的 `values(...)` 中增加明确的列别名，并保持轴顺序固定。
2. 如果字段属于奖励或 observation，扩展 `TrialRecord` 的数组字段。
3. 在 `_load_trial()` 中读取、同步并写入 `TrialRecord`。
4. 在 `_sample_record()` 中返回 `target_<name>` 和 `<name>_valid`。
5. 在 `source/isaaclab_assets/test/test_imitation_dataset.py` 增加一个小型 CSV 回归测试，验证字段名、单位和形状。
6. 在环境的 observation/reward term 中只消费 `target_<name>`，不要再做文件解析。

### 增加一个新的文件类型

文件解析应放在独立函数中，例如现有的 `_read_motion()`、`_read_pressure()` 和 `_load_event_file()`。函数应满足：

- 返回 NumPy 数组或 `None`，不返回 Isaac Sim 对象；
- 明确记录时间单位和空间单位；
- 对缺失或损坏的可选文件返回无效标志，而不是把随机值送入 PPO；
- 对必需的 IMU 数据抛出 `DatasetFormatError`，让 trial 被跳过或由 `strict=True` 直接失败。

### 修改任务过滤

不要在 `_discover_trials()` 中写死某个训练任务。更新 `task_mapping.yaml` 后，`load_task_mapping()` 会校验十个标准 ID。若需要新增任务，必须同时更新：

- `TASK_IDS`；
- `task_mapping.yaml`；
- 训练命令的任务选择器允许列表；
- 任务采样和回归测试。

## 10. 数据检查脚本

扫描全部任务并打印可用 trial、通道形状和有效标志：

```bash
uv run python - <<'PY'
from collections import Counter
from isaaclab_assets.tasks.imitation_dataset import ExoskeletonImitationDataset

root = "/home/lqf/MyCode/Data"
dataset = ExoskeletonImitationDataset(
    root,
    task_id="all",
    window_size=32,
    random_window=True,
    random_task_sampling=True,
    load_rgb=False,
    load_event=False,
)

print("trial_count:", len(dataset.trials))
print("task_count:", Counter(record.task_id for record in dataset.trials))
for task_id in dataset.task_ids:
    sample = dataset.sample(task_id=task_id, window_size=2)
    print(
        task_id,
        "imu", tuple(sample["target_imu"].shape),
        "imu_valid", sample["imu_valid"].tolist(),
        "motion_valid", bool(sample["motion_valid"]),
        "rgb_valid", sample["rgb_valid"].tolist(),
        "event_valid", bool(sample["event_valid"]),
    )
dataset.close()
PY
```

检查方向翻转是否只发生一次：

```bash
uv run python - <<'PY'
from isaaclab_assets.tasks.imitation_dataset import load_direction_calibration

mapping = load_direction_calibration("sensor_coordinate_mapping.yaml")
print("joint_sign:", mapping.joint_sign)
print("imu_orientation_sign:", mapping.imu_orientation_sign)
print("imu_gyro_sign:", mapping.imu_gyro_sign)
print("imu_accel_sign:", mapping.imu_accel_sign)
PY
```

检查单个 trial 的原始文件是否能被发现：

```bash
find /home/lqf/MyCode/Data -type d -name 'Trial*' | sort | head -n 20
find /home/lqf/MyCode/Data -type f \( -name '*_imu_0[1-6].csv' -o -name '*.aedat4' \) | sort | head -n 40
```

## 11. 常见错误和处理

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `no usable trials found` | 没找到 `Trial*`、任务 ID 不匹配，或没有 IMU CSV | 检查目录名、metadata 的 `task`、六路 IMU 文件 |
| `trial has no imu_01..imu_06 CSV files` | trial 缺少全部六路保留 IMU | 至少提供 IMU-01..IMU-06 中任一路；缺失的其他路会标记无效，但建议补齐六路 |
| `unknown task IDs` | 命令或 YAML 中使用了未注册的任务编号 | 只使用 BT01-BT05、AT01-AT05 |
| `motion_valid=False` | 没发现运动文件、形状不是 `[T,8]` 或有效样本少于 2 帧 | 使用包含 `motion`/`joint` 的文件名，检查八列和单位 |
| `rgb_valid=[False, False]` | 没安装解码器、视频路径错误或文件损坏 | 先用 `load_rgb=False` 验证 IMU，再安装 imageio/OpenCV 或修复 metadata |
| `event_valid=False` | 缺少 `dv-processing`、AEDAT4 损坏或使用了 preview MP4 | 安装解码器并使用原始 AEDAT4；不能用 RGB 差分替代 |
| 关节动作方向反了 | `joint_sign` 与现场编码器方向不一致 | 修改自定义 calibration YAML，重新训练 |
| IMU 数值量级异常 | deg/rad 或 dps/rad/s 列名不明确 | 使用带 `_deg`/`_rad` 和 `_dps`/`_rad_s` 的明确字段名 |
| 同一窗口每次都一样 | 固定了旧版本或没有递增内部采样计数 | 使用 `sample()`，不要缓存单个 sample；训练时调用 `set_epoch()` |

## 12. 修改后的验证顺序

每次修改读取器后按以下顺序验证：

1. 运行 Dataset 单元测试：

   ```bash
   uv run pytest -q source/isaaclab_assets/test/test_imitation_dataset.py
   ```

2. 运行离线扫描命令，确认 trial 数量没有意外下降。
3. 随机抽一个 trial，打印 `target_imu`、`t_s`、`imu_valid`、`motion_valid` 和事件有效标志。
4. 使用至少 16 个环境（例如 `--num_envs 16 --max_iterations 2`）做 Isaac Lab
   冒烟训练。当前 rl_games 配置为 `horizon_length=32`、`minibatch_size=512`，
   因此 1 个环境会产生 32 个 rollout 样本，触发
   `batch_size % minibatch_size == 0` 断言；单环境只用于 play/Viser 检查。
5. 方向校准或字段含义改变后，不要直接恢复旧 checkpoint；参考目标契约已经改变，应从新配置重新训练。

离线读取器不需要启动 Kit。只有当 Dataset 输出形状和有效标志确认正确后，才进入相机、IMU 传感器和 PPO 的端到端验证。
