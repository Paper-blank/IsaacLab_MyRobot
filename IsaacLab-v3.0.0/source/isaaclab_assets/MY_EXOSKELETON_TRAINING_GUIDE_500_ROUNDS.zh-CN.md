# 外骨骼 500 轮分阶段训练指南

本指南针对当前 `Isaac-MyExoskeleton-Flat-v0` 任务。这里的“轮”指 RL-Games 的 PPO `iteration`，不是单个物理仿真步。

当前 PPO 配置为 `horizon_length=32`，因此：

```text
1 iteration = 32 个控制步 = 32 * 0.02 s
500 iterations = 16,000 个控制步
```

课程已设置为每 500 个 PPO iteration 开放下一阶段。

| PPO iteration | 控制步 | 阶段 | 训练内容 |
| ---: | ---: | --- | --- |
| 0--499 | 0--15,999 | `WALK_ONLY` | 固定 `vx=0.2 m/s` 的 Walk |
| 500--999 | 16,000--31,999 | `TRANSITIONS` | Walk、W2S、Stand，速度上限 `0.8 m/s` |
| 1000 以后 | 32,000 以后 | `MULTI_GAIT` | Walk、Stand、W2S、Run、R2W，速度上限 `4.0 m/s` |

第二阶段不是 Stand-only：低速命令会先进入 W2S，低速且双脚支撑稳定 `1.5 s` 后才进入 Stand；Stand 收到非零速度命令后直接回到 Walk。论文没有独立的 Stand-to-Walk gait ID。

## 1. 训练前冒烟测试

所有命令在项目根目录执行：

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0

OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --num_envs=16 --headless --max_iterations=2
```

成功标准：没有 traceback；日志显示 actor `49` 维、critic `62` 维、动作 `8` 维，以及 `45` 个已注册 reward terms（未配置参考文件时，`sensor_reference_imitation` 的数值为零）。`rew_-inf` 文件名在只有 1 次 iteration、尚未结束完整 episode 时可以是正常现象。

## 2. 推荐训练命令

### 只验证第一阶段

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --num_envs=256 --headless --max_iterations=500
```

这条命令会在 Walk 阶段末尾结束。要观察阶段切换，建议让同一次运行继续跨过边界：

### 验证三个阶段切换

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --num_envs=256 --headless --max_iterations=1500
```

### 正式训练

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --num_envs=4096 --headless --max_iterations=30000
```

显存不足时将 `4096` 依次降为 `1024`、`256` 或 `64`。如果 Run 阶段在当前 8-DOF 外骨骼上不稳定，可以先把最终速度上限临时降到 `2--3 m/s`：

```bash
... env.curriculum.gait_commands.params.multi_gait_max_speed=2.5
```

## 3. 每个阶段的验收条件

不要只看总 reward，也不要只看自动切换的轮数。

### 第 500 轮前：Walk

- `error_vel_xy` 稳定低于 `0.05--0.10 m/s`。
- 固定 `0.2 m/s` 命令下实际前向速度明显大于零，目标是 `vx > 0.15 m/s`。
- `walk_contact` 持续上升，回放中能看到左右脚交替单支撑。
- `fallen` 接近零，`time_out` 占比上升。
- 不能只是站立、原地摆腿或双脚同时着地。

### 第 500--1000 轮：W2S/Stand

- 低速或零速度命令能经历 W2S，并在双脚支撑稳定后进入 Stand。
- 非零命令下 Stand 能重新进入 Walk，而不是一直保持站立。
- `w2s_contact`、`stand_contact`、`stand_base_motion` 等当前 gait 项只在对应 gait ID 下生效。

### 第 1000 轮以后：Run/R2W

- `run_contact` 能出现单支撑和短暂腾空，不应通过双脚静止获得高分。
- 降低速度时先进入 R2W，再平滑回到约 `0.2 m/s` Walk。
- 速度上限逐步增加；若跌倒率明显升高，先降低 `multi_gait_max_speed`，不要同时修改多个奖励项。

## 4. TensorBoard 监控

```bash
uv run tensorboard --logdir logs/rl_games/my_exoskeleton_ppo
```

重点查看：

```text
Episode/Curriculum/gait_commands/phase
Episode/Curriculum/gait_commands/control_step
Episode/Metrics/base_velocity/error_vel_xy
Episode/Metrics/base_velocity/error_vel_yaw
Episode/Episode_Termination/fallen
Episode/Episode_Termination/time_out
Episode/Episode_Reward/walk_contact
Episode/Episode_Reward/w2s_contact
Episode/Episode_Reward/stand_contact
Episode/Episode_Reward/run_contact
```

预期课程指标：第 500 轮附近 `phase` 从 `1` 变为 `2`，第 1000 轮附近变为 `3`。如果 Walk 指标没有达到验收条件，应将边界后移后重新训练，而不是把静止策略直接带入下一阶段。

## 5. 断点恢复

RL-Games checkpoint 不保存 `env.common_step_counter`。恢复时必须补回已经完成的 PPO iteration：

```text
step_offset = 已完成的 iteration 数 * 32
```

例如：

```bash
# 从第 500 轮继续
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=256 --headless \
  --max_iterations=1000 \
  --checkpoint /绝对路径/checkpoint.pth \
  env.curriculum.gait_commands.params.step_offset=16000

# 从第 1000 轮继续
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=256 --headless \
  --max_iterations=30000 \
  --checkpoint /绝对路径/checkpoint.pth \
  env.curriculum.gait_commands.params.step_offset=32000
```

这里的 `--max_iterations` 是本次运行结束时的累计 iteration 目标，不是从零开始的新增轮数；例如从第 500 轮恢复并填写 `1000`，表示再训练到第 1000 轮。

当前奖励、观测和状态机已经与旧实验不同。原来 200 轮的静止局部最优 checkpoint 不建议恢复；应从新配置开始训练。若修改了 `horizon_length`，必须同步重新计算边界和 `step_offset`。

## 6. 奖励与论文的关系

当前配置已经包含论文式 gait routing 和生物力学奖励：真实总质心角动量成本、左右腿镜像动量协调、腿部机械功率、双腿反相、幅值匹配、直膝支撑、摆脚高度、脚拖动、推蹬和接触相位奖励。

外骨骼没有论文中的手臂、腰和肩，因此手臂项使用明确标注的腿部形态适配代理，不能声称是手臂奖励的逐字复现。完整公式和权重见 [`MY_EXOSKELETON_REWARDS.zh-CN.md`](../../scripts/MY_EXOSKELETON_REWARDS.zh-CN.md)。

## 7. 结果位置与回放

训练结果位于：

```text
logs/rl_games/my_exoskeleton_ppo/<时间戳>/
```

训练结束后，用 `scripts/play_my_exoskeleton.py` 回放实际动作。回放时重点检查：机身是否直立、左右脚是否交替、Stand 是否能重新起步、减速是否平滑，以及是否出现原地摆腿或快速跌倒。

需要同时检查多模态输入时，可使用 Viser 和 JSONL 诊断：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data --num_envs 1 \
  --visualizer viser --diagnostics --diagnostics_interval 20 \
  --diagnostics-output logs/imitation_diagnostics.jsonl \
  --checkpoint /绝对路径/checkpoint.pth
```

诊断记录中的 `imu_valid_fraction`、`event_valid_fraction` 和 `imitation_reward` 用来区分传感器无效与策略本身没有学会。没有 EVIS `dvs_gen` 或 `dv-processing` 时，Event 通道必须保持无效，不能用 RGB 帧差代替。
