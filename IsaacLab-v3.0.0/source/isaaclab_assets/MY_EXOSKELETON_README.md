# my_exoskeleton 论文适配训练任务

本任务参考论文的多步态训练方法，结合当前外骨骼 URDF 和 Isaac Lab 接口进行了适配，并非逐项机械复刻。首次运行请先阅读[中文新手指南](MY_EXOSKELETON_BEGINNER_GUIDE.zh-CN.md)；各奖励项的公式和权重见 [`scripts/MY_EXOSKELETON_REWARDS.zh-CN.md`](../../scripts/MY_EXOSKELETON_REWARDS.zh-CN.md)，最终模型输出格式见 [`scripts/ruleresult.md`](../../scripts/ruleresult.md)，论文创新点与当前实现的逐项对应见 [MY_EXOSKELETON_PAPER_REPRODUCTION.zh-CN.md](MY_EXOSKELETON_PAPER_REPRODUCTION.zh-CN.md)。

## 当前实现

- 机器人有 8 个可控旋转关节，动作是按关节缩放的增量关节位置目标：髋转向 `0.45 rad`、髋屈伸 `0.65 rad`、膝 `0.95 rad`、踝 `0.50 rad`（左右腿同构）。这些是策略命令尺度，不是机械安全限位。
- 仿真步长为 `0.005 s`，`decimation=4`，策略控制周期为 `0.02 s`。
- 速度目标每 `3-6 s` 重采样；步态状态机以 `10 Hz` 更新。
- Isaac 导入和 articulation 层均开启自碰撞；由于当前高密度 STL 凸包有相邻重叠，必须同时检查碰撞体排斥造成的姿态偏移。
- 关节、IMU 和策略时序契约见 [`23solve.md`](../../23solve.md)；修改动作尺度后，旧 checkpoint 不应继续使用，必须重新训练。
- actor 的原生动作仍为 8 维位置目标；回放启用 `--ruleresult` 后，可输出 24 维的角度、角速度和角加速度结果，格式与实测状态说明见 [`scripts/ruleresult.md`](../../scripts/ruleresult.md)。
- 一个 episode 最长 `20 s`，姿态倾角超过 `65 deg` 时提前终止。
- 启动时随机化地面摩擦、基座质量和 PD 增益；训练中每 `3-6 s` 注入横向速度和偏航扰动。

核心文件：

- `source/isaaclab_assets/envs/my_exoskeleton_env_cfg.py`：场景、观测、动作、45 个已注册奖励项（其中传感器模仿项默认无参考数据时为零）、随机化、终止条件和课程配置。
- `source/isaaclab_assets/tasks/gait.py`：五类步态及三阶段课程的共享定义。
- `source/isaaclab_assets/tasks/my_exoskeleton_flat.py`：命令状态机、观测和奖励实现。
- `source/isaaclab_assets/configs/my_exoskeleton_ppo.yaml`：循环 actor 和集中式 critic 的 PPO 配置。
- `scripts/train_my_exoskeleton.py`：训练入口。

## 观测与网络

Actor 接收 49 维观测：

```text
projected gravity                         3
base angular velocity                    3
base linear velocity                     3
joint position / velocity / previous action 24
left/right contact magnitude             2
left/right pressure mean/std              4
velocity command                          3
gait phase sin/cos                        2
gait one-hot                              5
                                         --
total                                    49
```

Critic 接收 62 维特权状态，即 actor 的 49 维观测，再增加双脚接触标志 `2`、归一化接触力 `2`、根节点高度 `1` 和关节加速度 `8`。

PPO 使用单层 `64` 单元 LSTM actor，后接 `[32]` ELU MLP；集中式 critic 使用独立的前馈 `[32]` ELU MLP。当前 RL-Games 版本未给 central critic 配置循环层，以避免 rollout 时隐藏状态被重复推进。主要参数为 `horizon_length=32`、`seq_length=16`、`minibatch_size=512`、初始动作标准差 `0.8`、`entropy_coef=0.01` 和最多 `30000` 个 epoch。

## 五步态与课程

策略接收以下固定顺序的五维 one-hot 标签：

```text
0 Stand
1 Walk
2 Walk-to-Stand
3 Run
4 Run-to-Walk
```

三阶段课程按全局控制步推进：

| 阶段 | 控制步 | 训练内容 |
| --- | ---: | --- |
| Walk only | `0-15999` | 前 500 个 PPO iteration，固定 `0.2 m/s` 前向行走 |
| Transitions | `16000-31999` | 第 501--1000 个 PPO iteration，速度上限 `0.8 m/s`，加入站立和 Walk-to-Stand |
| Multi-gait | `>=32000` | 第 1001 个 iteration 起，速度上限 `4.0 m/s`，启用全部五种步态 |

以 `horizon_length=32` 计算，当前两个阶段边界对应第 `500` 和 `1000` 个 PPO iteration。Run 使用 `Fr=v^2/(g*0.8)>0.5` 判定；Walk-to-Stand 需要低速、双脚支撑且实际速度低于 `0.1 m/s` 持续 `1.5 s`；Run-to-Walk 持续 `2.5 s` 后切换为 Walk。

论文没有单独的 Stand-to-Walk gait。Stand 收到非零命令后会直接切为 Walk，由循环策略学习起步过程。它是需要单独回放验收的隐式过渡，不应误称为论文定义的第六种 gait。

## 奖励适配

Reward Manager 共注册 45 项：19 项共享奖励/惩罚（其中传感器模仿项没有参考文件时为零），以及 Walk `4` 项、Stand `8` 项、Walk-to-Stand `4` 项、Run `6` 项和 Run-to-Walk `4` 项。步态专属项通过 gait mask 路由，同一环境只激活当前步态对应的一组。

共享项覆盖速度跟踪、存活/终止、机身姿态与高度、关节速度/加速度、动作变化、关节限位和动量协调；步态项覆盖相位接触、摆脚高度、拖脚、双脚支撑、静止姿态、短接触、蹬伸和减速过渡等目标。

当前模型没有手臂和腰部。总质心角动量使用各刚体质量、惯量、质心位置与速度真实计算；论文公式中的左右手臂差值改用显式分组的左右腿角动量。手臂能耗、对称和幅值奖励分别用腿部机械功率、双腿反相和双腿幅值匹配替代，且协调正奖励只有机器人实际前进时才生效。肘、腰和肩奖励没有对应自由度，未伪造实现。

Phase 1 额外使用静止时严格为零的前向速度窗口，低速线速度跟踪也要求正向运动；Walk 接触必须完整匹配左右单支撑模式，直膝只约束支撑腿。这些约束用于避免“站住或原地摆腿也能拿高分”的局部最优。模型没有全身碰撞传感器，因此使用非超时终止惩罚作为碰撞/跌倒代理。Isaac Lab ContactSensor 提供聚合接触力，而不是论文中的足底压力阵列；观测中使用左右脚接触力及其均值/标准差代理。

## 运行训练

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=4096 --headless --max_iterations=30000
```

先做小规模冒烟测试：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=16 --headless --max_iterations=1
```

默认任务 ID 为 `Isaac-MyExoskeleton-Flat-v0`，日志写入 `logs/rl_games/my_exoskeleton_ppo/`。

## Checkpoint 兼容性

旧任务使用不同的观测接口和前馈策略；当前任务使用 49 维循环 actor、62 维集中式 critic，并更改了奖励、命令状态机和课程。因此旧 checkpoint 在结构和训练语义上都不兼容，必须从头训练，不能用于恢复当前任务。

只有由当前 49D/62D 配置和当前奖励语义生成的 checkpoint 才应恢复；恢复时应向训练入口传入一个实际存在的 checkpoint 文件路径。RL-Games checkpoint 不保存环境的课程计数，因此还要设置：

```text
env.curriculum.gait_commands.params.step_offset=<已完成iteration数*32>
```

例如从第 500 个 iteration 恢复时使用 `step_offset=16000`，从第 1000 个 iteration 恢复时使用 `step_offset=32000`，否则环境课程会重新从 Phase 1 计数。

## 当前验证状态

- Python 编译检查通过。
- Kit-less 回归共 `92 passed`（完整 Kit/PhysX 配置测试另行运行），覆盖步态枚举、Froude 阈值、500/1000 轮课程边界、接触读取、惯量坐标变换、低速跟踪门控、Run 腾空/W2S 双支撑、gait mask 奖励路由、传感器参考数据、多模态数据集、固定相机、事件通道和非循环目标序列。
- RL-Games 模型构建检查通过：actor 输入 `49`、输出 `8`，LSTM hidden/cell 为 `(1, 16, 64)`，初始 sigma 约为 `0.8`；central critic 输入 `62`、输出 `(16, 1)`。
- 真实 Isaac Sim/RL-Games 冒烟训练会验证 policy `(49,)`、critic `(62,)` 和 `45` 个已注册奖励项；该检查只代表训练链路可运行。

上述冒烟测试只验证环境、循环策略、集中式 critic 和 PPO 更新链路能够运行，不代表策略已经收敛。正式使用前仍需执行完整课程训练并评估站立、行走、跑步和两种过渡步态。
