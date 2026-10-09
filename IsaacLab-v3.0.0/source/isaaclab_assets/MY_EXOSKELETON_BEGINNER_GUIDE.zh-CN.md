# 外骨骼 Isaac Lab 项目新手指南

这份指南面向第一次接触本项目的人，带你完成下面这条完整路径：

```text
拿到完整项目 -> 配好运行环境 -> 小规模启动 -> 查看训练结果 -> 修改模型或训练逻辑 -> 再次验证
```

本项目不是网页或 ROS 节点工程。它是一个基于 **Isaac Lab + Isaac Sim + RL-Games PPO** 的外骨骼强化学习训练项目：在仿真中控制 8 个关节，用同一个循环策略学习站立、行走、跑步以及两种减速过渡。奖励项的完整说明见 `scripts/MY_EXOSKELETON_REWARDS.zh-CN.md`。

> 所有命令都应在项目根目录执行：`/home/lqf/MyCode/IsaacLab-v3.0.0`。下面的命令使用 `uv` 管理 Python 环境；不要用裸 `python`、不要手动 `pip install`，也不需要激活 conda 环境。

## 1. 先了解项目会做什么

当前任务的 Gym ID 是 `Isaac-MyExoskeleton-Flat-v0`，训练脚本是 `scripts/train_my_exoskeleton.py`。它会自动完成任务注册并默认选择 RL-Games，因此日常运行都从这个脚本开始。

| 项目内容 | 当前配置 |
| --- | --- |
| 机器人 | 8 个可控旋转关节，另有 1 个固定关节 |
| 动作 | 8 维增量关节位置目标 |
| Actor 观测 | 49 维：本体、关节位置/速度、上一时刻归一化动作、接触/命令观测 + 步态相位 `sin/cos` + 5 维 gait ID |
| Critic 观测 | 62 维：49 维 actor 观测 + 训练期可见的接触、力、根高度和关节加速度 |
| 物理与仿真 | Isaac Sim 6.0.1 / PhysX，GPU 运行 |
| 算法 | RL-Games PPO；单层 64 单元 LSTM 后接 32 单元 MLP，asymmetric actor-critic |
| 输出位置 | `logs/rl_games/my_exoskeleton_ppo/<时间戳>/` |

训练不是从第一步就混合全部动作模式，而是按全局环境控制步逐级开放：前 500 个 PPO iteration（`0--15999` 控制步）只训练 Walk，第 501--1000 个 iteration（`16000--31999`）加入 Stand 与 W2S，第 1001 个 iteration 起（`>=32000`）再加入 Run 与 R2W。五个 gait ID 同时用于 actor 条件输入和逐环境奖励掩码；当前模式不相关的 gait 专属奖励为零。阶段边界的内部单位仍是控制步，不是 PPO epoch，完整规则和权重见奖励说明文档。

当前 `horizon_length=32`，所以第 200 个 PPO iteration 只累计约 `6400` 个控制步，仍处于 Walk-only；Phase 2 在第 500 个 iteration、Phase 3 在第 1000 个 iteration 开始。论文没有独立的 Stand-to-Walk 标签：Stand 收到非零命令后直接切换到 Walk，策略在 Walk 条件下学习从静止起步。

## 2. 开始前的条件

### 2.1 机器条件

完整的 Isaac Sim 工作流需要 Linux x86_64、Ubuntu 22.04 或更高版本、可用的 NVIDIA 显卡与驱动。项目使用 Python 3.12；`uv` 会自动创建对应环境，不需要单独安装或激活 Python 虚拟环境。

官方最低建议是 32 GB 内存和 16 GB 显存。第一次运行请从 16 个并行环境开始，**不要直接使用 4096**。本项目当前机器的 RTX 4090 24 GB 已验证过 4096 环境；其他显卡应逐步增加数量。

先在终端确认显卡和驱动可见：

```bash
nvidia-smi
```

如果这条命令失败，先修复 NVIDIA 驱动并重启，再继续本指南。系统 CUDA Toolkit 不是本项目的前置条件。

### 2.2 必须拿到这个完整项目目录

这个目录包含定制的外骨骼 URDF、网格、任务代码和 PPO 配置，不是仅靠上游 Isaac Lab 仓库就能还原的。迁移到新电脑时，请完整复制当前项目目录，至少保留：

```text
pyproject.toml
uv.lock
scripts/train_my_exoskeleton.py
source/isaaclab_assets/data/my_robot/
source/isaaclab_assets/envs/my_exoskeleton_env_cfg.py
source/isaaclab_assets/tasks/my_exoskeleton_flat.py
source/isaaclab_assets/tasks/gait.py
source/isaaclab_assets/isaaclab_assets/my_exoskeleton.py
source/isaaclab_assets/configs/my_exoskeleton_ppo.yaml
```

不要把本项目的 `uv` 训练环境和 Docker/ROS/Gazebo 环境混用。这里的训练依赖 Isaac Sim 6.0.1，而项目中的 Docker 配置服务于另一条工作流。

## 3. 第一次配置环境

### 3.1 安装 `uv`

先检查是否已经安装：

```bash
uv --version
```

若终端提示找不到命令，在 Linux 上运行下面的官方安装命令，然后重新打开一个终端：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果刚安装完、当前终端仍找不到 `uv`，执行：

```bash
source "$HOME/.local/bin/env"
```

### 3.2 创建项目环境并安装必需组件

进入项目根目录后，运行一次以下命令：

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
uv sync --locked --extra isaacsim --extra rl-games
```

它会根据项目的 `uv.lock` 创建 `.venv` 并安装精确锁定的依赖，包括 Isaac Sim、RL-Games 与 CUDA 版 PyTorch。首次下载量较大，请确保网络可访问 PyPI、NVIDIA Python 源和 GitHub，并预留充足磁盘空间。

`--locked` 的含义是严格使用项目已经验证过的依赖版本，而不改动锁文件。除非改了 `pyproject.toml`、`uv.lock` 或更换了电脑，通常不需要重复执行这一步。

### 3.3 验证安装但不启动仿真

下面的命令只显示训练脚本帮助，不会开始长时间训练：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --help
```

正常结果是看到 `Train an RL agent with RL-Games.` 和参数列表。终端可能同时出现 Gym 已停止维护的提示；只要命令成功输出帮助且退出码为 0，这只是上游依赖的兼容性警告，不是本项目失败。

也可以验证自定义任务已注册：

```bash
uv run --extra isaacsim --extra rl-games python -c "import gymnasium as gym; import isaaclab_assets.my_exoskeleton; print(gym.spec('Isaac-MyExoskeleton-Flat-v0').id)"
```

预期输出：

```text
Isaac-MyExoskeleton-Flat-v0
```

## 4. 第一次真正运行：小规模无界面冒烟训练

首次运行使用 16 个并行环境和 2 次训练迭代。它足够验证 URDF、传感器、奖励、PPO 和日志输出是否连通，又不会像正式训练那样占用大量显存。

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=16 --headless --max_iterations=2
```

这里的几个参数含义如下：

| 参数 | 含义 |
| --- | --- |
| `--num_envs=16` | 同时运行 16 个相互独立的外骨骼仿真副本 |
| `--headless` | 不打开图形窗口，适合服务器和节省显存；本项目脚本会将它转换为 Isaac Lab 3 的 `--visualizer none` |
| `--max_iterations=2` | 只训练 2 个迭代，用于快速验证 |
| `OMNI_KIT_ACCEPT_EULA=YES` | 非交互方式接受 NVIDIA Omniverse EULA，避免首次启动等待输入 |

第一次加载 Isaac Sim 扩展和资源缓存可能较慢，尤其是新机器。请先等待终端持续输出初始化或训练信息，而不是几秒后就判断它卡住了。

成功后会创建类似下面的目录：

```text
logs/rl_games/my_exoskeleton_ppo/2026-09-09_02-42-44/
├── nn/          # 模型 checkpoint，后缀为 .pth
├── params/      # 本次运行冻结下来的环境与 PPO 配置
├── summaries/   # TensorBoard 事件文件
└── run.json     # 任务和训练库的基本信息
```

> 不要在这个脚本后面追加 `--rl_library rl_games`。该参数不属于此封装脚本的命令行接口，脚本本身已经固定默认使用 RL-Games。

## 5. 想看见机器人时怎么运行

在有本地图形桌面的机器上，先只启动 1 个环境，并显式使用 Kit 可视化器：

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=1 --max_iterations=1 \
  --visualizer kit --max_visible_envs=1 \
  agent.params.config.minibatch_size=32 \
  agent.params.config.central_value_config.minibatch_size=32
```

这里两个 `minibatch_size=32` 覆盖都很重要：actor PPO 与 privileged critic 各有自己的
minibatch 配置。默认值 `512` 是为 16 或 4096 个环境准备的，而 1 个环境每轮只有
`32（horizon_length）× 1 = 32` 个样本。RL-Games 要求两个 batch 都能被各自的 minibatch 整除。
如果不想手动覆盖 minibatch，也可以把 `--num_envs=1` 改成 `--num_envs=16`，这正好使用默认的 512。

这条命令适合检查机器人是否悬空、穿地、关节方向是否正确，以及双脚是否接触地面。图形窗口的启动并不等于策略已经学会走路：第 1 次迭代的动作仍接近随机。

无显示器的服务器、远程 SSH 或窗口启动失败时，回到第 4 节的 `--headless` 命令即可。不要为此强行配置 X11；本项目的日常训练不需要图形窗口。

## 6. 正式训练、日志和恢复

### 6.1 逐级扩大训练规模

在 16 环境冒烟成功前，不要开始长训练。推荐按下面顺序尝试；每次只提高一个档位，发生显存不足就退回上一档：

```bash
# 小规模调试
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=256 --headless

# 显存充足时的目标规模
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=4096 --headless --max_iterations=30000
```

不传 `--max_iterations` 时，PPO YAML 中的默认上限为 30000 个 epoch。训练时在另一个终端运行 `nvidia-smi`，观察显存是否稳定。如果出现 CUDA out of memory，结束当前进程并降低 `--num_envs`，例如从 4096 降到 1024 或 256。

不要只按轮数机械进入下一阶段。Phase 1 至少应同时满足：`error_vel_xy` 稳定低于 `0.05--0.10 m/s`、`walk_contact` 明显高于 0、`fallen` 接近 0，并在回放中看到真正的交替落脚。Phase 2 再分别回放 Walk→W2S→Stand 和 Stand→Walk；这些通过后才评估 Phase 3 的 Run/R2W。若第 500 轮仍未达到这些指标，应把边界后移，而不是直接相信自动切换结果。

### 6.2 查看 TensorBoard

训练输出生成后，在项目根目录启动 TensorBoard：

```bash
uv run tensorboard --logdir logs/rl_games/my_exoskeleton_ppo
```

终端会显示本地访问地址，通常是 `http://localhost:6006`。在浏览器中打开它，可以查看 reward、loss、学习率和吞吐量。按 `Ctrl+C` 停止 TensorBoard，不会删除训练结果。

### 6.3 恢复训练

恢复时使用明确的 checkpoint 文件，并补回环境课程计数。RL-Games 会恢复网络和优化器，但不会保存 `env.common_step_counter`：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=4096 --headless \
  --checkpoint /绝对路径/last_my_exoskeleton_ppo_ep_500_rew_xxx.pth \
  env.curriculum.gait_commands.params.step_offset=16000
```

`step_offset = 已完成的 iteration 数 * 32`。例如从第 200 轮恢复对应 `6400`，从第 500 轮恢复对应 `16000`，从第 1000 轮恢复对应 `32000`。不设置时，新环境会从 Phase 1 重新计数。修改了关节数、动作维度或观测维度后，旧 checkpoint 通常不再兼容，应重新训练。

本次任务把 actor 输入从 42 维改为 49 维，并加入 LSTM、privileged critic、五类 gait ID 和新的奖励路由。最新生物力学修改还改变了奖励语义；旧的 200 轮静止策略不应继续训练，应从新配置新建实验。

### 6.4 判断模型是否学会，以及打开可视化回放

可以打开 Kit 窗口查看训练后的动作，但“窗口能打开”只说明仿真启动成功，不能说明策略已经学会。建议先看 TensorBoard，再用 1 个环境回放同一个 checkpoint。

#### 第一步：看训练指标

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
uv run tensorboard --logdir logs/rl_games/my_exoskeleton_ppo
```

浏览器打开 `http://localhost:6006`，重点观察下面这些曲线：

| 指标 | 训练正常时的方向 | 含义 |
| --- | --- | --- |
| `episode_lengths/iter` | 接近最大回合长度（本任务约 1000 步） | 机器人能持续站立/运动，而不是马上结束回合 |
| `Episode/Episode_Termination/fallen` | 接近 0 | 摔倒次数少 |
| `Episode/Episode_Termination/time_out` | 接近 1 | 多数回合是走完整个时限后正常结束 |
| `Episode/Metrics/success_rate` | 持续上升并稳定 | 速度指令跟踪成功率；当前 XY 阈值已收紧为 `0.05 m/s` |
| `Episode/Metrics/base_velocity/error_vel_xy`、`Episode/Metrics/base_velocity/error_vel_yaw` | 逐步下降 | 实际速度与目标速度的误差 |
| `Episode/Curriculum/gait_commands/phase` | 第 500/1000 轮变为 2/3 | 当前课程阶段 |
| `Episode/Episode_Reward/walk_contact` | Phase 1 中上升 | 完整单支撑模式匹配；双支撑得 0。Run 另有小段腾空窗口，W2S 允许末端双支撑 |
| `rewards/iter` | 上升后趋于稳定 | 只能作为综合信号，不能单独判断成功 |

如果 `rewards/iter` 上升，但 `episode_lengths/iter` 约等于 1、`fallen` 接近 1、`time_out` 接近 0，通常是机器人第一步就摔倒，模型还不合适。

#### 第二步：用 Kit 回放 checkpoint

训练进程必须先结束，或者至少不要在同一时刻写入正在读取的 checkpoint；否则可能读到半写入文件并与训练争抢显存。训练终端按 `Ctrl+C` 停止后，运行：

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py \
  --checkpoint /home/lqf/MyCode/IsaacLab-v3.0.0/logs/rl_games/my_exoskeleton_ppo/2026-09-21_20-34-51/nn/my_exoskeleton_ppo.pth \
  --num_envs=1 --visualizer kit --max_visible_envs=1 --real-time
```

上面的 `2026-09-21_12-46-27` 是当前已有的一次训练目录；使用其他训练结果时，只替换这段目录名和 `.pth` 文件名，不要保留尖括号。也可以用 `--checkpoint best` 自动选择奖励最高的模型，用 `--checkpoint latest` 选择最新模型。注意 `best` 是按 reward 选的，不保证就是动作表现最好的模型；本任务应结合下面的摔倒率和回放画面判断：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py --checkpoint best \
  --num_envs=1 --visualizer kit --max_visible_envs=1 --real-time
```

#### 在 Kit 中查看可点击的 IMU 数据

`Visualization Markers` 原本只控制可视化开关，不能读取 IMU 数值；普通 marker 是批量实例，Kit 无法分别选中其中某一个。回放时追加 `--show-imu`，会为每个 IMU 创建独立的蓝色可点击探针：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py --checkpoint best \
  --num_envs=1 --visualizer kit --max_visible_envs=1 --real-time --show-imu
```

`--show-imu` 只能与 `--num_envs=1 --visualizer kit` 一起使用。启动后，在 Kit 右侧的 `IsaacLab` 面板中展开 `Visualization Markers`，六个 IMU 项目已启用；也可以直接在视口中点击蓝色探针。选中探针后，在 `Property` 面板的自定义 `imu:*` 属性中查看：

- `imu:angularVelocity`：IMU 本体坐标系角速度，单位 `rad/s`。
- `imu:linearAcceleration`：IMU 本体坐标系线加速度，单位 `m/s^2`。
- `imu:sourcePrim`：该传感器绑定的机器人链接路径。

数值约每 0.1 秒刷新一次。探针只是调试几何体，不参与碰撞、不改变策略观测或已训练的 checkpoint。

回放时按下面几项判断：机器人能否保持躯干直立、脚是否交替接触地面、动作是否连续而不高频抖动、是否跟随目标速度前进、是否很快翻倒。至少观察多个回合，不能只看刚启动的几秒。

按 `Ctrl+C` 退出回放。若要录制约 40 秒（1200 步）视频，需要额外启用项目的 `video` 依赖：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games --extra video \
  python scripts/play_my_exoskeleton.py --checkpoint best \
  --num_envs=1 --visualizer kit --max_visible_envs=1 --real-time \
  --video --video_length 1200 --video_interval 1
```

视频通常写入对应训练目录的 `videos/play/`。无显示器时可用 `--headless --video` 录制，不会弹出窗口。

### 6.5 可选：同步到 Weights & Biases

已完成 `wandb login` 的情况下，可追加：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=16 --headless --max_iterations=2 \
  --track True --wandb-project-name my-exoskeleton \
  --wandb-entity your-wandb-user-or-team
```

将 `your-wandb-user-or-team` 替换为自己的 W&B 用户名或团队名。`--track True` 必须和
`--wandb-entity` 一起提供；不需要在线实验跟踪时，TensorBoard 已足够，省去账号和网络依赖。

## 7. 文件地图：想改什么就去哪里

| 目标 | 文件 | 改完后最先做什么 |
| --- | --- | --- |
| 机器人连杆、关节限位、惯量、传感器定义 | `source/isaaclab_assets/data/my_robot/urdf/my_exoskeleton.urdf` | 先用 1 环境图形模式检查模型 |
| 机器人外形网格 | `source/isaaclab_assets/data/my_robot/meshes/` | 重新启动后检查模型显示与碰撞 |
| URDF 路径、初始高度、PD、电机、传感器路径、速度命令、动作幅度、奖励权重、终止条件 | `source/isaaclab_assets/envs/my_exoskeleton_env_cfg.py` | 16 环境冒烟训练 |
| 49/62 维 actor/critic 观测、步态状态机、接触特征、奖励公式 | `source/isaaclab_assets/tasks/my_exoskeleton_flat.py` | 语法检查，再做 16 环境冒烟训练 |
| gait ID、Froude 阈值和 curriculum 常量 | `source/isaaclab_assets/tasks/gait.py` | 运行 gait 单元测试 |
| 任务名和 Gym 注册 | `source/isaaclab_assets/isaaclab_assets/my_exoskeleton.py` | 用第 3.3 节的注册检查确认 ID |
| PPO 网络与超参数 | `source/isaaclab_assets/configs/my_exoskeleton_ppo.yaml` | 先运行少量迭代，观察日志 |
| 自动注册与启动参数兼容 | `scripts/train_my_exoskeleton.py` | 先运行 `--help` |

运行结果位于 `logs/`，生成的 USD 文件和 `__pycache__/` 也不是应手改的源码。尤其不要直接编辑 `source/isaaclab_assets/data/my_robot/urdf/my_exoskeleton/` 下的 `.usda` 或 `.usd` 文件；它们是 URDF 导入过程产生的内容，机器人结构的唯一源应当是 `.urdf` 和 `meshes/`。

## 8. 最常见的修改方式

### 8.1 调整初始姿态、PD 参数和动作幅度

编辑 `source/isaaclab_assets/envs/my_exoskeleton_env_cfg.py`：

- `ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01))`：根初始高度。当前 URDF 的脚底在根坐标系中接近 `z=0`，因此用约 1 cm 的安全间隙贴近平面；换用其他模型时必须重新测量。
- `MyExoskeletonEventsCfg.reset_scene_to_default`：episode 终止后恢复根位姿、速度和关节状态。只配置 `fallen` 终止而没有 reset event，会让倒地状态继续留在物理仿真中。
- `MyExoskeletonRewardsCfg.termination_penalty`：对倒地等非超时终止施加负奖励；其余项分为所有模式共享的 regulation/task reward 和逐 gait 掩码项。
- `ImplicitActuatorCfg(... stiffness=80.0, damping=4.0)`：PD 执行器刚度和阻尼。先小幅调整一次，不要同时大改二者。
- `JointPositionActionCfg(... scale=JOINT_ACTION_SCALE_RAD)`：策略每步的关节位置增量范围按关节设置（髋转向 `0.45 rad`、髋屈伸 `0.65 rad`、膝 `0.95 rad`、踝 `0.50 rad`）。这些是策略尺度，不等于硬件限位；完整迁移约束见 [`23solve.md`](../../23solve.md)。
- `MyExoskeletonCurriculumCfg.gait_commands`：在 16k/32k 控制步边界（约第 500/1000 个 PPO iteration）扩展前进、横移与偏航命令范围，并开放新的 gait。
- `env.curriculum.gait_commands.params.step_offset`：断点恢复时补回已完成的环境控制步，通常等于恢复 iteration 数乘 32。
- `MyExoskeletonEventsCfg`：启动时随机化摩擦、基座质量和执行器增益，训练中按区间施加横向推扰。

每次改完先用第 5 节的 1 环境图形模式确认没有明显穿地或抖动，再跑第 4 节的 16 环境冒烟训练。

### 8.2 修改奖励或终止条件

奖励的两部分必须配套：

1. 在 `source/isaaclab_assets/tasks/my_exoskeleton_flat.py` 编写或修改计算函数。每个函数应返回每个环境一个值，即形状为 `(num_envs,)` 的张量。
2. 在 `source/isaaclab_assets/envs/my_exoskeleton_env_cfg.py` 的 `MyExoskeletonRewardsCfg` 中，用 `RewardTermCfg` 配置该函数和权重。

例如，函数本身通常返回一个正的量，而 `weight` 决定它是奖励还是惩罚；论文 Eq. (2) 的动量项是例外，它直接返回非正的平方成本并使用正权重。Walk、Stand、W2S、Run、R2W 的专属函数还必须保留逐环境 `gait_mask`，否则互斥目标会在同一步同时生效。修改时一次只改变一个奖励项或一个权重，记录本次实验目录，才容易判断效果来自哪里。

跌倒判断位于 `MyExoskeletonTerminationsCfg`，当前姿态阈值为 65 度。把阈值调得更严格会让回合更早结束；调得更宽松则可能允许不稳定姿态继续累积奖励。

### 8.3 增减观测或动作

观测和动作有明确的维度契约，不能只改一处：

- Actor 当前 49 维，其中最后 7 维是相位 `sin/cos` 与五类 gait one-hot；critic 当前 62 维，并且只能在训练期使用额外的 privileged 特征。
- 新增或删除观测特征时，同时更新 `my_exoskeleton_flat.py` 中 actor/critic 的拼接逻辑、环境配置中的维度声明、RL-Games observation-group 映射和文件中的维度注释。
- 改变可控关节时，同时更新 URDF、`ACTUATED_JOINT_NAMES`、`action_space`，以及 PPO 配置的说明。
- 关节顺序必须一致。动作数组的第 `i` 个元素会按 `ACTUATED_JOINT_NAMES` 的顺序控制对应关节。

旧 checkpoint 的网络输入输出尺寸固定。任何维度变化后都应视为新实验，而不是继续加载旧模型。

### 8.4 更换 URDF、关节或网格

按下面顺序改，排错最快：

1. 修改 `my_exoskeleton.urdf` 或 `meshes/`，确认 URDF 中的网格相对路径仍正确。
2. 若关节或链接改名，更新 `ACTUATED_JOINT_NAMES`，并同步修改 `my_exoskeleton_env_cfg.py` 中 IMU 和左右脚接触传感器的 `prim_path`。
3. 检查初始高度、关节限位、PD 参数和动作幅度是否仍适合新结构。
4. 先运行第 5 节的 1 环境图形模式，再运行第 4 节的 16 环境冒烟训练。

URDF 导入会生成 USD 缓存。当前外骨骼配置固定使用 `force_usd_conversion=True`，因此修改 URDF、碰撞类型或自碰撞设置后会按当前配置重建缓存。若你在自定义配置中关闭了该字段，请先确认已经停止旧训练进程，再临时开启并运行一次重新导入；不要直接修改生成的 USD 文件来绕过这个问题。

### 8.5 调整 PPO 超参数

编辑 `source/isaaclab_assets/configs/my_exoskeleton_ppo.yaml`：

- `learning_rate`：学习率，当前为 `3.0e-4`。
- `horizon_length`：每个环境每次采样的步数，当前为 `32`。
- `minibatch_size` 与 `mini_epochs`：每批训练的拆分方式和重复优化次数。
- `network.rnn`：actor 的单层 LSTM 配置；当前隐藏宽度为 64。
- `network.mlp.units`：LSTM 后策略网络的隐藏层宽度；当前为 `[32]`。
- `central_value_config`：只在训练期使用 62 维 privileged observation 的前馈 critic 配置。当前 RL-Games 版本下有意不为 central critic 增加 RNN。
- `seq_length`：循环 PPO 的序列长度。它必须整除 `horizon_length` 和两个 minibatch。
- `max_epochs`：不传 `--max_iterations` 时的训练上限。

`--num_envs` 是日常控制并行仿真数量的入口，优先在命令行调整它。超参数修改后先使用 `--num_envs=16 --max_iterations=2` 检查是否能正常启动，再开始长训练。

## 9. 修改后的固定验证流程

这个项目没有网页开发中的热更新。训练或 Isaac Sim 已经运行时，文件改动不会自动生效；先停止当前进程，再重新启动。

每次改动后按下面顺序验证：

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0

# 1. 先检查常改 Python 文件是否有语法错误
uv run python -m py_compile \
  scripts/train_my_exoskeleton.py \
  source/isaaclab_assets/envs/my_exoskeleton_env_cfg.py \
  source/isaaclab_assets/tasks/my_exoskeleton_flat.py \
  source/isaaclab_assets/tasks/gait.py \
  source/isaaclab_assets/isaaclab_assets/my_exoskeleton.py

# 2. 机器人、关节、初始姿态或传感器改动后，先用 1 个环境可视化检查
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=1 --max_iterations=1 \
  --visualizer kit --max_visible_envs=1 \
  agent.params.config.minibatch_size=32 \
  agent.params.config.central_value_config.minibatch_size=32

# 3. 任何改动都至少做一次无界面小规模训练验证
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=16 --headless --max_iterations=2
```

项目的 workspace 包以 editable 方式安装。修改上述 Python、URDF、STL 或 YAML 文件后，不需要重新 `pip install` 或执行 `colcon build`；停止进程并重新运行即可。只有改动依赖定义（例如 `pyproject.toml`）时，才重新执行：

```bash
uv sync --locked --extra isaacsim --extra rl-games
```

## 10. 常见问题

### `uv: command not found`

安装 `uv` 后重新打开终端，或执行 `source "$HOME/.local/bin/env"`。确认 `uv --version` 能输出版本后再继续。

### `No module named isaacsim` 或 `No module named rl_games`

通常是遗漏了可选组件。回到项目根目录执行：

```bash
uv sync --locked --extra isaacsim --extra rl-games
```

之后的训练命令也保留 `--extra isaacsim --extra rl-games`，这样命令本身就完整表达了所需环境。

### `nvidia-smi` 不可用、CUDA 初始化失败或显存不足

前两种情况先检查 NVIDIA 驱动、系统版本和 GPU 是否满足 Isaac Sim 要求。显存不足时不要改一堆 PPO 参数，首先把 `--num_envs` 降到 16、64、256 或 1024，并使用 `--headless`。

### 图形窗口打不开

服务器或 SSH 会话通常没有图形桌面。使用 `--headless` 是正常做法；只有需要人工检查机器人时才在本地桌面使用 `--visualizer kit`。

### 图形窗口显示“无响应”或启动很久没有画面

第一次使用 `--visualizer kit` 时，Isaac Sim 可能要编译 RTX Shader、加载扩展并建立缓存。窗口在这段时间内可能被桌面标记为“无响应”，但进程仍在工作；本项目一次实测 Shader 编译约 129 秒、完整启动约 206 秒。首次启动时先等待 3--5 分钟，不要连续启动多个实例。后续启动会复用缓存，通常明显更快。

可以在另一个终端观察进程和显卡是否仍在工作：

```bash
pgrep -af 'train_my_exoskeleton|isaacsim|kit'
nvidia-smi
```

如果能看到训练进程，且 `nvidia-smi` 中显存或 GPU 利用率有变化，通常不是死机。日志位于
`.venv/lib/python3.12/site-packages/isaacsim/kit/logs/Kit/IsaacLab/3.0/`；查看最新日志末尾：

```bash
tail -f "$(ls -t .venv/lib/python3.12/site-packages/isaacsim/kit/logs/Kit/IsaacLab/3.0/kit_*.log | head -1)"
```

日常训练建议直接使用 `--headless`，这样不会等待或依赖图形窗口。只有在本地桌面检查机器人模型时才使用 Kit 可视化器。

### `AssertionError`，指向 batch、minibatch 或 sequence length

这是 RL-Games PPO 的批次大小约束，不是显卡故障。每轮样本数为
`num_envs * horizon_length`；当前 `horizon_length=32`、默认 `minibatch_size=512`，所以
`--num_envs=1` 时得到 `32 % 512 != 0`。循环策略还要求 `seq_length` 整除
`horizon_length`、actor minibatch 和 critic minibatch。单环境可视化命令末尾加入：

```text
agent.params.config.minibatch_size=32 \
agent.params.config.central_value_config.minibatch_size=32
```

或者把环境数改为 16（`16 * 32 = 512`），继续使用默认 minibatch。`--max_iterations=1` 只限制训练轮数，不会改变这个整除关系。

### 看到 `rew_-inf` 或大量 `Failed to find rigid body/contact report API`

单次 `--max_iterations=1` 的冒烟运行可能生成 `rew_-inf` 文件名，因为还没有任何完整回合可用于计算最佳回报；这不等于训练失败。Isaac Sim 的 URDF 导入器还可能为嵌套的 link/geometry prim 打印 `Failed to find rigid body/contact report API`。这些警告本身不能证明接触传感器失效：本任务的 ContactSensor 仍应解析出左右脚 body ID，并提供非零的 `data.net_normal_forces_w`。

判断传感器是否真的失效，以三项为准：启动过程是否出现 traceback、Reward Manager 是否列出 45 个已注册奖励项、短时运行中左右脚接触力是否有非零值。只有接触力始终为零或出现 traceback 时，才检查
`source/isaaclab_assets/envs/my_exoskeleton_env_cfg.py` 中 IMU、左右脚接触传感器的 `prim_path`，并与 URDF 导入后的链接名称逐字核对。这类警告与首次 Kit Shader 编译造成的“无响应”是两个独立问题。

### 显示 `Gym has been unmaintained...`

这是 RL-Games 依赖发出的兼容性提示。若后面仍正常显示参数、初始化和训练信息，可以忽略；若命令随即报出其他异常，应以最后的 traceback 为准。

### `Task ... not found`

请使用本项目的 `python scripts/train_my_exoskeleton.py ...`，而不是直接执行通用的 `uv run isaaclab train ...`。前者会导入 `isaaclab_assets.my_exoskeleton` 并注册自定义任务。

### 改完 Python 或 URDF，结果看起来没有变化

确认旧进程已经退出，再重新运行。Python、URDF 和网格不会被运行中的训练热加载；URDF 还可能需要按第 8.4 节重新导入 USD 缓存。

## 11. 每天开始工作的最短清单

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
nvidia-smi
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=16 --headless --max_iterations=2
```

这三步成功后，再按目标修改参数、扩大 `--num_envs` 或延长训练。这样每一次实验都有一个可靠的起点。
