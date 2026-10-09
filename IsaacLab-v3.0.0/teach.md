# 外骨骼模仿强化学习训练指令

本文给出当前项目从启动、加载环境和真实数据集，到训练、回放和检查传感器的完整命令。所有命令都在项目根目录执行：

传感器开关、左右腿 IMU 轴矩阵和常用组合命令的集中版见
[`sensor_switch_and_training_tutorial.md`](sensor_switch_and_training_tutorial.md)。

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
```

项目使用 Isaac Lab 3.0 manager-based API 和 `rl_games` PPO。训练入口会自动导入
`isaaclab_assets.my_exoskeleton`，因此不需要手动调用旧版 Isaac Gym 注册接口。

## 0. 最短启动流程

如果只需要马上开始一次真实数据模仿训练，按下面顺序执行。第一次建议保留
`--num_envs 16 --max_iterations 2` 做冒烟检查；确认没有错误后，再改成正式配置。

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
export OMNI_KIT_ACCEPT_EULA=YES
export DATASET_ROOT=/home/lqf/MyCode/Data

# 先验证数据集可以被读取（不启动 Isaac Sim）
uv run python - <<'PY'
import os

from isaaclab_assets.tasks.imitation_dataset import ExoskeletonImitationDataset

dataset = ExoskeletonImitationDataset(
    os.environ["DATASET_ROOT"], task_id="BT02", window_size=32,
    random_window=True, load_rgb=False, load_event=False,
)
print("usable trials:", len(dataset.trials), "tasks:", dataset.task_ids)
dataset.close()
PY

# 启动 Isaac Lab 环境并载入 BT02 数据
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 16 --task_id BT02 \
  --imitation_dataset "$DATASET_ROOT" \
  --headless --max_iterations 2

# 冒烟训练通过后，使用 4096 个并行环境正式训练
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 4096 --task_id BT02 \
  --imitation_dataset "$DATASET_ROOT" \
  --headless --max_iterations 30000
```

其中 `--task` 是 Gym 环境 ID，`--task_id` 是数据集动作 ID，
`--imitation_dataset` 是数据集根目录。三者分别负责“载入哪一个环境”“采样哪一种动作”和
“从哪里读取记录”，不能把 `BT02` 写到 `--task` 中。

## 1. 环境与依赖

确认 Isaac Sim、Isaac Lab 和 `rl_games` 的工作区依赖可用：

```bash
uv run python -c "import isaaclab, isaaclab_rl, gymnasium; print('Isaac Lab environment is ready')"
uv run python -c "import isaaclab_assets.my_exoskeleton as t; print(t.IMITATION_TASK_ID)"
```

正式训练需要 Kit/PhysX 和 GPU。推荐使用项目锁定环境并接受 Isaac Sim 许可：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

如果 `uv` 环境没有安装 Isaac Sim extra，请先执行：

```bash
uv sync --extra isaacsim --extra rl-games
```

## 2. 可用环境和任务

Gym 环境 ID：

```text
Isaac-MyExoskeleton-Flat-v0       传统平面速度跟踪环境
Isaac-MyExoskeleton-Imitation-v0  Vision + IMU + Motion Imitation 环境
```

真实记录任务由 `task_mapping.yaml` 定义：

| 任务 ID | 含义 |
| --- | --- |
| `BT01` | 静态站立标定 |
| `BT02` | 自然行走 |
| `BT03` | 慢速行走 |
| `BT04` | 快速行走 |
| `BT05` | 起步—行走—停止 |
| `AT01` | 转身 |
| `AT02` | 障碍跨越 |
| `AT03` | 斜坡行走 |
| `AT04` | 台阶行走 |
| `AT05` | 手提重物行走 |

`--task` 选择 Gym 环境，`--task_id` 选择数据集任务；两者不要混用。例如，`BT02`
必须配合 `Isaac-MyExoskeleton-Imitation-v0` 使用。

训练脚本还接受以下等价写法：

| 用途 | 推荐参数 | 兼容别名 |
| --- | --- | --- |
| 数据集动作 | `--task_id BT02` | `--task-id BT02` |
| 数据集路径 | `--imitation_dataset /path/to/Data` | `--dataset`、`--imitation-dataset` |
| 无图形界面 | `--headless` | 等价转换为 `--visualizer none` |

不要同时提供同一类参数的不同值；脚本会在启动前报告冲突，而不会静默选择其中一个。

确认环境已经注册（这一步只读取 Gym spec，不创建 Kit 仿真实例）：

```bash
uv run python - <<'PY'
import gymnasium as gym
import isaaclab_assets.my_exoskeleton  # registers both environments

for task in ("Isaac-MyExoskeleton-Flat-v0", "Isaac-MyExoskeleton-Imitation-v0"):
    spec = gym.spec(task)
    print(task, "->", spec.entry_point)
PY
```

真正的环境实例由 `train_my_exoskeleton.py` 或 `play_my_exoskeleton.py` 在 Isaac Sim
启动后创建；不要在普通 Python 进程中直接 `gym.make` 来替代 Kit 启动器。

## 3. 先检查数据集

默认数据集目录为 `/home/lqf/MyCode/Data`。先只扫描 IMU，不启动仿真：

```bash
uv run python - <<'PY'
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
print(f"trials={len(dataset.trials)} tasks={dataset.task_ids}")
for task_id in dataset.task_ids:
    sample = dataset.sample(task_id=task_id, window_size=2)
    print(
        task_id,
        "imu=", tuple(sample["target_imu"].shape),
        "motion_valid=", bool(sample["motion_valid"]),
        "imu_valid=", bool(sample["imu_valid"].all()),
    )
dataset.close()
PY
```

当前记录通常包含六路 IMU、RGB 视频和 AEDAT4 事件文件。缺少 `joint_pos_rad` 时，
`motion_valid=False`，关节角模仿项会自动屏蔽；这不是程序把 IMU 姿态误当作关节角。
要启用关节奖励，需要提供 `[T,8]` 的弧度轨迹，顺序必须是：

```text
base_to_Rturn, Rturn_to_Rthigh, Rthigh_to_Rleg, Rleg_to_Rfoot,
base_to_Lturn, Lturn_to_Lthigh, Lthigh_to_Lleg, Lleg_to_Lfoot
```

AEDAT4 解码需要 `dv-processing`，仿真异步事件需要 EVIS 的 `dvs_gen`。缺少这些可选组件
时，event 通道会标记为无效，绝不会用 RGB 帧差伪造事件。

### 数据目录约定

数据集根目录应保持如下结构（目录名中的任务 ID 必须是 `task_mapping.yaml` 中的十个 ID
之一；元数据 JSON 中的 `task` 字段优先级更高）：

```text
/home/lqf/MyCode/Data/
└── records/<subject>/<date>/<task_id>/TrialXX/
    ├── <prefix>_imu_01.csv ... <prefix>_imu_06.csv   # 必需
    ├── left_daheng_recording.mp4                     # 可选 RGB
    ├── right_daheng_recording.mp4                    # 可选 RGB
    ├── left_dvxplorer_recording.aedat4               # 可选 Event
    ├── right_dvxplorer_recording.aedat4              # 可选 Event
    ├── <motion>.csv/.npz/.pt                         # 可选 [T, 8] joint_pos_rad
    └── *.json                                        # 可选元数据
```

数据加载器以 50 Hz（`target_dt_s=0.02`）对六路 IMU 和可用媒体做时间同步，并在 reset 时
按 `window_size=32` 取随机窗口。`--task_id all` 会在有效 trial 中随机抽取任务；指定
`BT02`、`BT03` 等单个 ID 时只加载该任务。若某个 trial 缺少全部六路保留 IMU，它会被跳过并打印
警告；只缺少部分通道时，缺失路会标记为无效并填零。可用 trial 数为零时训练会直接失败，
这时先修复数据目录或元数据。

若要启用可选通道，可安装对应依赖：

```bash
# Viser 回放（只在使用 --visualizer viser 时需要）
uv sync --extra viser

# AEDAT4 事件解码（版本需与当前 Python/Isaac 环境兼容）
uv pip install dv-processing
```

### 方向校准（已在读取层启用）

数据加载器默认读取项目根目录的
[`sensor_coordinate_mapping.yaml`](sensor_coordinate_mapping.yaml)，在同步和采样前完成
方向修正：

| 通道 | 默认处理 |
| --- | --- |
| 髋/膝关节方向（canonical 索引 `1,2,5,6`） | 按镜像 joint axis 使用 `[+1,-1,-1,+1]`（右髋、右膝、左髋、左膝）；速度同步取同一符号 |
| IMU-01/03/05 左腿（01/03/05 分别为踝/膝/髋） | `[x_d,y_d,z_d] -> [-y_d,x_d,-z_d]` |
| IMU-02/04/06 右腿（02/04/06 分别为踝/膝/髋） | `[x_d,y_d,z_d] -> [y_d,x_d,-z_d]` |
| IMU-07..11 | 忽略，不进入观测或奖励 |

如果传感器安装方向不同，可复制该 YAML 并通过 Hydra 覆盖：

```bash
... env.imitation_calibration_path=/absolute/path/my_sensor_coordinate_mapping.yaml
```

当前数据目录扫描结果中多数 trial 没有 `[T,8]` 的 `joint_pos_rad` 文件，因此关节符号修正
会在检测到 motion 文件后生效；IMU 修正对现有六路 CSV 立即生效。该处理只校准参考数据，
不会修改 Isaac Sim 的关节轴或动作控制器；若使用没有关节参考数据的纯速度训练，仍需另外
检查 URDF 关节轴和动作符号。方向校准会改变模仿奖励的目标契约，启用后应从新配置重新
训练，不要直接把旧 checkpoint 当作已校准策略继续使用。

角速度和加速度的读取链为 `R_exo = R_joint_link @ R_link_sensor @ R_sensor_IMU`。
由于当前数据没有传感器安装四元数，`R_joint_link` 与 `R_link_sensor` 默认是单位阵，不能把它们
当作已经完成实体安装标定；现场测量后在自定义 YAML 中按 `imu_01`--`imu_06` 覆盖。姿态默认
保持 `euler_components` 兼容模式；只有有效矩阵为右手旋转时才使用 `rotation_matrix` 姿态模式。

## 4. 第一次冒烟训练

先用 16 个环境运行 2 个 PPO iteration，检查 USD 导入、传感器路径、数据同步和奖励形状：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 16 \
  --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data \
  --headless \
  --max_iterations 2
```

没有 traceback 且能生成 `logs/rl_games/` 下的训练目录，才进入正式训练。GPU 显存不足时，
先保持相同配置把 `--num_envs` 降到 `8` 或 `16`；PPO 配置的正式并行数仍是 4096。

## 5. 正式训练单个任务

自然行走（BT02）：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 4096 \
  --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data \
  --headless \
  --max_iterations 30000
```

慢速行走（BT03）：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 --num_envs 4096 \
  --task_id BT03 --imitation_dataset /home/lqf/MyCode/Data \
  --headless --max_iterations 30000
```

快速行走（BT04）：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 --num_envs 4096 \
  --task_id BT04 --imitation_dataset /home/lqf/MyCode/Data \
  --headless --max_iterations 30000
```

静态站立标定：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 4096 --task_id BT01 \
  --imitation_dataset /home/lqf/MyCode/Data \
  --headless --max_iterations 30000
```

BT01 的零速任务在所有课程阶段都保持 `GaitMode.STAND`，不会被第一阶段的固定前向行走项
重新改成 Walk。

## 6. 多任务混合训练

使用 `all` 时，环境在每次 reset 为环境随机抽取一个有效任务窗口：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 4096 \
  --task_id all \
  --imitation_dataset /home/lqf/MyCode/Data \
  --headless --max_iterations 30000
```

建议先分别让 BT01--BT05 收敛，再加入 AT01--AT05。当前场景按要求仍是 2.4 m 平面直线通道；
AT02--AT04 的数据标签会参与模仿，但不会自动生成障碍、斜坡或台阶几何。

## 7. 500 iteration 课程

PPO 配置使用 `horizon_length=32`，所以课程控制步边界如下：

| PPO iteration | 阶段 |
| ---: | --- |
| `0--499` | 固定 `0.2 m/s` Walk |
| `500--999` | 加入 Walk-to-Stand 和 Stand，速度上限 `0.8 m/s` |
| `1000+` | 开放 Walk、Stand、W2S、Run、Run-to-Walk，速度上限 `4.0 m/s` |

这里的 500 是 PPO 的课程阶段边界，不是数据集任务切换器：运行 `--max_iterations 500`
不会自动从 BT02 切换到 BT03。需要切换动作任务时，重新指定 `--task_id`，或使用
`--task_id all` 做多任务随机采样；需要延续同一策略时使用 checkpoint 和下面的
`step_offset` 恢复方式。

验证课程边界可运行 1500 iteration：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data \
  --num_envs 256 --headless --max_iterations 1500
```

从旧 checkpoint 恢复时，RL-Games 不保存环境课程步数。若 checkpoint 已完成 `N` 个
iteration，必须加：

```bash
env.curriculum.gait_commands.params.step_offset=$((N * 32))
```

例如从第 500 轮继续到第 1000 轮：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data \
  --num_envs 4096 --headless --max_iterations 1000 \
  --checkpoint /absolute/path/checkpoint.pth \
  env.curriculum.gait_commands.params.step_offset=16000
```

旧的 200 轮站立 checkpoint 使用了不同的观察/奖励契约，不建议直接续训；正式实验应从
当前配置重新开始。

## 8. 播放 checkpoint

单环境 Kit 回放：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data \
  --num_envs 1 \
  --visualizer kit \
  --checkpoint /absolute/path/checkpoint.pth
```

检查六路 IMU 时追加 `--show-imu`（必须同时是 `--num_envs 1 --visualizer kit`）：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data --num_envs 1 \
  --visualizer kit --show-imu \
  --checkpoint /absolute/path/checkpoint.pth
```

Viser 和 JSONL 诊断：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data --num_envs 1 \
  --visualizer viser --diagnostics --diagnostics_interval 20 \
  --diagnostics-output logs/imitation_diagnostics.jsonl \
  --checkpoint /absolute/path/checkpoint.pth
```

诊断字段包括 `imu_valid_fraction`、`camera_rgb_features`、`event_valid_fraction` 和
`imitation_reward`。`event_valid_fraction=0` 表示 EVIS 未安装或没有有效事件流，不表示
系统偷偷使用了 RGB 差分。

## 9. 自碰撞确认

当前 URDF 导入和 PhysX articulation 两层均已开启自碰撞，并设置了
`force_usd_conversion=True`。修改 URDF 或碰撞网格后首次启动会重新生成 USD。可以先用
以下 Kit-less 检查确认配置：

```bash
uv run python - <<'PY'
from isaaclab_assets.envs.my_exoskeleton_env_cfg import MyExoskeletonFlatEnvCfg

cfg = MyExoskeletonFlatEnvCfg()
spawn = cfg.scene.robot.spawn
print("self_collision:", spawn.self_collision)
print("enabled_self_collisions:", spawn.articulation_props.enabled_self_collisions)
print("force_usd_conversion:", spawn.force_usd_conversion)
print("collision_type:", spawn.collision_type)
PY
```

Kit 场景中还应看到 `physxArticulation:enabledSelfCollisions=True`。如果视觉上出现连杆
被推开的“形变”，先检查相邻 STL 凸包是否重叠；当前配置使用凸分解碰撞体，不能替代低模
碰撞网格和相邻 link 碰撞过滤。

## 10. 验证与常见问题

运行无 Kit 回归测试：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run pytest -q \
  source/isaaclab_assets/test --ignore=source/isaaclab_assets/test/test_valid_configs.py
uv run ruff check source/isaaclab_assets scripts/train_my_exoskeleton.py scripts/play_my_exoskeleton.py
```

如果打开 checkpoint 后一直重复同一个起步或 W2S 动作，先确认 checkpoint 是否只训练了
前 500 iteration、播放时命令是否一直为零、LSTM 是否在 episode 边界清零，以及是否使用了
同一版本的 observation normalization。BT05 和高级任务的目标序列默认在末帧保持，不会自动
从停止动作跳回起点；周期性 BT02/BT03/BT04 才允许窗口循环。

完整 Kit/PhysX reset、相机渲染和 EVIS 事件流必须在可用 GPU 的 Isaac Sim 运行环境中验证；
上述 Kit-less 测试不能替代真实长时间训练或实体机安全测试。
