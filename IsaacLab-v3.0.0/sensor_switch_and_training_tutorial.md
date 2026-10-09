# 外骨骼传感器开关与训练教程

本文说明 `Isaac-MyExoskeleton-Imitation-v0` 的可选传感器开关、训练与回放命令、
观测/奖励契约，以及真实数据读取时的 IMU 左右腿坐标校准。

## 1. 环境和数据集

训练入口使用 Isaac Lab 3.0 manager-based API 和 `rl_games` PPO：

```text
Isaac-MyExoskeleton-Flat-v0       传统平面速度跟踪环境
Isaac-MyExoskeleton-Imitation-v0  Vision + IMU + Motion Imitation 环境
```

本教程的传感器开关适用于 `Isaac-MyExoskeleton-Imitation-v0`。数据集默认路径为：

```text
/home/lqf/MyCode/Data
```

`--task` 选择 Gym 环境，`--task_id` 选择数据集动作。可用任务由
[`task_mapping.yaml`](task_mapping.yaml) 定义：

| ID | 动作 |
| --- | --- |
| `BT01` | 静态站立标定 |
| `BT02` | 自然行走 |
| `BT03` | 慢速行走 |
| `BT04` | 快速行走 |
| `BT05` | 起步、行走、停止 |
| `AT01` | 转身 |
| `AT02` | 障碍跨越 |
| `AT03` | 斜坡行走 |
| `AT04` | 台阶行走 |
| `AT05` | 手提重物行走 |

使用 `--task_id all`（大小写均可）时，每次 reset 从有效任务窗口中随机采样；指定
`BT02` 等单个 ID 时只使用该动作。默认窗口为 32 帧、50 Hz（约 0.64 s）。

## 2. 传感器开关

四个开关都默认启用。参数可以写成等号形式或空格形式：

```bash
--imu=0       # 等价于 --imu 0
--rgb=1       # 等价于 --rgb 1
--event=true
--contact=false
```

也接受 `0/1`、`true/false`、`yes/no`、`on/off`，不区分大小写。省略某个开关时，
配置中的默认值 `True` 生效。

| 开关 | 覆盖字段 | 覆盖范围 |
| --- | --- | --- |
| `--imu` | `env.use_imu` | IMU-01 到 IMU-06、IMU 观测、IMU/gyro 奖励 |
| `--rgb` | `env.use_rgb` | 左右 RGB 相机、RGB 特征和 RGB 奖励 |
| `--event` | `env.use_event` | 左右 Event Camera、EVIS runtime 和 event 奖励 |
| `--contact` | `env.use_contact` | 左右足接触传感器、接触观测和接触相关奖励 |

开关在 Kit 启动前会被入口脚本转换成 Hydra 覆盖，例如：

```text
--imu=0     -> env.use_imu=False
--rgb=1     -> env.use_rgb=True
```

传感器关闭时有三层行为：

1. 对应的 scene entity 不创建，避免分配不需要的 IMU、RTX 或 EVIS 缓冲区。
2. observation 仍保持原来的固定形状，返回全零张量，并将 `*_valid` 标记为 `False`。
3. 对应 imitation/contact reward 返回零或从有效通道归一化分母中移除，不会把缺失传感器
   当成完美匹配。

RGB/Event 的关闭覆盖会在 `launch_simulation()` 扫描配置之前把对应 camera entity 置为
`null`，所以 `--rgb=0 --event=0` 不会启动 RTX 相机渲染；如果任一视觉通道仍开启，
Isaac Lab 会根据 `CameraCfg` 自动设置 `enable_cameras=True`。这也是 headless 训练仍能
读取 RGB 的原因。没有 EVIS `dvs_gen` 插件时，Event Camera 只返回无效零事件，绝不会用
RGB 帧差伪造事件；不需要事件流时建议显式使用 `--event=0`。

因此关闭传感器不会改变 PPO 网络输入维度，但会改变输入语义；建议用与训练相同的开关
回放 checkpoint。不要把 `--imu` 单独写成无值选项，必须提供 `0/1` 或布尔字符串。

`--contact=0` 适合做动力学消融和排错；它会同时移除足底接触传感器、置零接触观测并跳过
接触奖励，因此依赖支撑相/摆动相时序的步态训练不应把它作为最终配置。

### 常用组合

```bash
# 全部启用：也可以完全省略四个参数
--imu=1 --rgb=1 --event=1 --contact=1

# 只用 IMU 和本体状态
--imu=1 --rgb=0 --event=0 --contact=0

# IMU + RGB，不启动 Event Camera
--imu=1 --rgb=1 --event=0 --contact=1

# 动力学/控制调试：关闭四类可选传感器
--imu=0 --rgb=0 --event=0 --contact=0
```

## 3. 训练命令

先用少量环境做冒烟检查：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 16 \
  --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data \
  --imu=1 --rgb=0 --event=0 --contact=1 \
  --headless --max_iterations 2
```

确认 USD、数据集和 observation 形状无误后，再启动 4096 并行训练：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 4096 \
  --task_id all \
  --imitation_dataset /home/lqf/MyCode/Data \
  --headless --max_iterations 30000
```

例如只训练自然行走、关闭事件流：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 4096 --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data \
  --imu=1 --rgb=1 --event=0 --contact=1 \
  --headless --max_iterations 30000
```

如果本机不是通过 `uv` 管理环境，可将命令中的 `uv run ... python` 替换为项目 Isaac
环境中的 `python`；必须保证 Isaac Sim、Isaac Lab、`rl_games` 和数据集解码依赖来自同一
环境。

### 500 轮课程边界

PPO 配置的 `horizon_length=32`，当前课程按 PPO iteration 划分：

| iteration | 课程 |
| ---: | --- |
| `0--499` | 固定前进速度的 Walk 阶段 |
| `500--999` | 加入 Walk-to-Stand/Stand，速度上限约 `0.8 m/s` |
| `1000+` | 开放 Walk、Stand、Run 和过渡阶段 |

`--max_iterations 500` 不会自动从 BT02 切换到 BT03；任务切换由 `--task_id` 或
`--task_id all` 控制。恢复 checkpoint 时补回课程步数，例如已完成 500 轮：

```bash
... --checkpoint /absolute/path/checkpoint.pth \
    env.curriculum.gait_commands.params.step_offset=16000
```

传感器开关、动作顺序、坐标校准或 observation 语义改变后，旧 checkpoint 不应直接作为
正式实验初始化权重；至少先用匹配开关回放确认输入契约。

## 4. Play / Viser 回放

单环境 Kit 回放（推荐先用这个检查模型动作）：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 1 --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data \
  --imu=1 --rgb=1 --event=0 --contact=1 \
  --visualizer kit \
  --checkpoint /absolute/path/checkpoint.pth
```

检查六路 IMU 探针时追加 `--show-imu`。该选项必须同时满足 `--num_envs 1` 和
`--visualizer kit`：

```bash
... --num_envs 1 --visualizer kit --show-imu \
    --checkpoint /absolute/path/checkpoint.pth
```

Viser 和 JSONL 诊断：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 1 --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data \
  --imu=1 --rgb=1 --event=1 --contact=1 \
  --visualizer viser --diagnostics --diagnostics_interval 20 \
  --diagnostics-output logs/imitation_diagnostics.jsonl \
  --checkpoint /absolute/path/checkpoint.pth
```

诊断快照包含任务计数、`imu_valid_fraction`、RGB 特征/原图尺寸、
`event_valid_fraction` 和 imitation reward 统计。`event_valid_fraction=0` 通常表示 EVIS
不可用或事件文件未成功解码；实现不会用 RGB 帧差伪造 Event Camera。

## 5. 观测和奖励契约

策略观测使用固定的 manager group，便于 4096 环境 PPO 和 checkpoint 复用：

| group / key | 形状（每环境） | 内容 |
| --- | ---: | --- |
| `policy/state` | `49` | 基础姿态、速度、8 关节位置/速度、上一动作、足底接触、命令和 gait 状态 |
| `imu` / `obs["imu"]` | `6 x 9 = 54` | 每路 roll/pitch/yaw、gyro xyz、加速度 xyz；manager 输入展平为 54 |
| `camera_rgb` / `obs["camera_rgb"]` | `12` | 左右相机的 RGB 均值和标准差特征；原图仅保留在 play 诊断缓存 |
| `event_camera` / `obs["event_camera"]` | `32` | 两相机、正负 polarity、每相机 8 个空间池化 bin |
| `critic/state` | `62` | policy state 加接触类别、接触力、根部高度和关节加速度 |

完整 actor 输入为 `49 + 54 + 12 + 32 = 147`。禁用任意可选传感器时，54、12 或 32
维仍保留为零，因此不会因为切换开关而改变网络层尺寸；有效性 mask 只影响数据和奖励。

模仿奖励使用正的 `exp(-MSE / sigma^2)` 相似度，并按有效通道归一化：

| 项 | 默认权重 | 比较内容 |
| --- | ---: | --- |
| `imitation_imu` | `1.0` | IMU 姿态、角速度、加速度 |
| `imitation_gyro` | `0.5` | 六路陀螺仪 |
| `imitation_joint` | `0.5` | 八个 canonical joint position |
| `imitation_velocity` | `0.5` | 根部平面速度和任务速度 |
| `imitation_rgb` | `0.25` | 两路 RGB 特征 |
| `imitation_event` | `0.25` | 事件 histogram 特征 |

基础站立、步态、生物力学、能耗和安全项仍由平面环境 reward 配置提供。没有
`[T, 8]` `joint_pos_rad` 的 trial 会将 `motion_valid=False`，关节模仿项自动屏蔽；
要训练关节角模仿，数据必须使用 canonical 顺序并提供弧度值。

## 6. IMU 映射与方向校准

加载器只保留 IMU-01 到 IMU-06，忽略 IMU-07 到 IMU-11。默认 link 映射见
[`imu_mapping.yaml`](imu_mapping.yaml)：

| 传感器 | 侧别 | robot link | 用途 |
| --- | --- | --- | --- |
| IMU-01 | left | `left_foot` / `Lfoot` | 左足踝 |
| IMU-02 | right | `right_foot` / `Rfoot` | 右足踝 |
| IMU-03 | left | `left_shank` / `Lleg` | 左膝 |
| IMU-04 | right | `right_shank` / `Rleg` | 右膝 |
| IMU-05 | left | `left_thigh` / `Lthigh` | 左髋 |
| IMU-06 | right | `right_thigh` / `Rthigh` | 右髋 |

实际 USD link 名称会在环境创建后自动匹配；需要人工确认时设置：

```bash
... env.imu_mapping_output_path=/absolute/path/imu_mapping_resolved.yaml
```

### 左右腿轴矩阵

[`sensor_coordinate_mapping.yaml`](sensor_coordinate_mapping.yaml) 规定矩阵约定为
`output = matrix @ dataset_vector`，并对同一路 IMU 的 orientation、gyro、accel 使用同一
左右腿轴变换：

```text
left  = [[ 0, -1,  0],
         [ 1,  0,  0],
         [ 0,  0, -1]]

right = [[ 0,  1,  0],
         [ 1,  0,  0],
         [ 0,  0, -1]]
```

因此 IMU-01/03/05 使用 `left`，IMU-02/04/06 使用 `right`；IMU-07..11 忽略。
默认数据集读取路径只应用上面的轴矩阵，不再叠加旧的全局取负。下面的 `*_sign` 字段
仅为未传 `sensor_index` 的外部旧调用保留，不能在奖励函数或控制器中再次应用。

```text
orientation: [+1, -1, +1]
gyro:        [-1, -1, +1]
accel:       [-1, -1, +1]
```

关节参考数据的 canonical 顺序为：

```text
base_to_Rturn, Rturn_to_Rthigh, Rthigh_to_Rleg, Rleg_to_Rfoot,
base_to_Lturn, Lturn_to_Lthigh, Lthigh_to_Lleg, Lleg_to_Lfoot
```

读取层默认关节符号为 `[+1, +1, -1, +1, +1, -1, +1, +1]`：右髋/右膝/左髋/左膝
使用 `[+1,-1,-1,+1]` 的镜像轴符号，转向和踝关节保持。这个修正发生在数据同步/采样前，
不会修改 USD 关节轴；如果现场 IMU
安装方向不同，复制 YAML 后覆盖：

```bash
... env.imitation_calibration_path=/absolute/path/my_sensor_coordinate_mapping.yaml
```

## 7. 相机和 Event Camera 检查

两侧相机是每个并行环境内的固定设备，不是随机生成的 Viser 摄像机模型：

| 设备 | 位置 |
| --- | --- |
| left RGB/Event | `(x=1.2, y=+1.2, z=1.0) m` |
| right RGB/Event | `(x=1.2, y=-1.2, z=1.0) m` |

两侧均 look-at 通道中心 `(1.2, 0, 0.7) m`，RGB 分辨率为 `256 x 256`。通道长度为
`2.4 m`，左右边界几何位于 `y=+/-1.2 m`。配置检查（不创建仿真）可使用：

```bash
python - <<'PY'
from isaaclab_assets.envs.my_exoskeleton_imitation_env_cfg import MyExoskeletonImitationEnvCfg

cfg = MyExoskeletonImitationEnvCfg()
print("num_envs:", cfg.scene.num_envs)
print("rgb:", cfg.scene.left_rgb_camera.width, cfg.scene.left_rgb_camera.height)
print("channel:", cfg.scene.channel_length_m, cfg.scene.channel_half_width_m)
print("sensors:", cfg.use_imu, cfg.use_rgb, cfg.use_event, cfg.use_contact)
PY
```

Viser/Newton 默认显示机器人和通道几何，不会把 RGB/Event sensor prim 当作可见的摄像机
模型；首次打开网页还要等待第一个仿真帧并展开模型树。要检查相机 prim、实时 RGB 画面和
安装方向，优先使用 `--visualizer kit`；Viser 的相机流需要另外启用
`streaming_view=True` 并指定已有相机 prim，不能仅凭 Viser 中没有“摄像机外形”判断传感器
未创建。

## 8. 回归验证

入口参数和传感器禁用行为的无 Kit 测试：

```bash
python -m pytest -q source/isaaclab_assets/test/test_play_my_exoskeleton.py
python -m pytest -q source/isaaclab_assets/test/test_my_exoskeleton_imitation_runtime.py
python -m pytest -q source/isaaclab_assets/test/test_my_exoskeleton_imitation_config.py
ruff check scripts/train_my_exoskeleton.py scripts/play_my_exoskeleton.py \
  source/isaaclab_assets/test/test_play_my_exoskeleton.py
```

数据集离线扫描（不启动 Kit）：

```bash
python - <<'PY'
from isaaclab_assets.tasks.imitation_dataset import ExoskeletonImitationDataset

dataset = ExoskeletonImitationDataset(
    "/home/lqf/MyCode/Data",
    task_id="all",
    window_size=32,
    random_window=True,
    random_task_sampling=True,
    load_rgb=False,
    load_event=False,
)
print("trials:", len(dataset.trials), "tasks:", dataset.task_ids)
for task_id in dataset.task_ids:
    sample = dataset.sample(task_id=task_id, window_size=2)
    print(task_id, tuple(sample["target_imu"].shape), "motion_valid=", bool(sample["motion_valid"]))
dataset.close()
PY
```

若 `event_valid_fraction` 为零，先检查 EVIS/dv-processing 和 AEDAT4 文件；若
`motion_valid=False`，先补齐 `[T,8]` 关节轨迹。传感器开关改变后若出现 Hydra unknown key，
确认使用的是本项目的 imitation Gym task，而不是旧入口或旧 Flat 配置。
