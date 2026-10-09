# 外骨骼多步态奖励与训练策略

## 实现范围

当前任务参考论文 *Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion*，用一个策略学习五种模式：

| gait ID | 模式 |
| ---: | --- |
| `0` | Stand |
| `1` | Walk |
| `2` | Walk-to-Stand (W2S) |
| `3` | Run |
| `4` | Run-to-Walk (R2W) |

论文机器人具有手臂、腰部和躯干关节，本项目的外骨骼只有 8 个腿部自由度。因此 elbow phase、waist/shoulder control 等项无法直接计算；不会为不存在的关节伪造信号。`leg_momentum_balance` 使用 Isaac Lab 的刚体质量、世界系惯量、质心位置和速度计算真实总质心角动量，再用左右腿角动量替代论文公式中的左右手臂项。arm energy、arm symmetry 和 arm-leg amplitude match 分别适配为腿部机械功率、双腿反相和双腿幅值匹配。当前模型也没有全身碰撞传感器，所以仍用非超时终止惩罚近似论文的 collision penalty。

## 观测与 gait 路由

Actor 输入为 49 维：本体状态、关节位置/速度、上一时刻归一化动作、接触和命令观测，加上步态相位的 `sin/cos` 两维和 gait ID one-hot 五维。训练期 critic 输入为 62 维，在 actor 输入之外加入双脚接触标志、归一化接触力、根高度和 8 维关节加速度；这些 privileged 特征不会送入部署时的 actor。

共享奖励始终生效。模式专属奖励在函数内部乘逐环境 gait mask：只有当前 gait ID 对应的环境会获得该项，其他环境该项严格为零。这样可以避免站立、行走、跑步和过渡目标同时相互干扰。

步态分类器以 10 Hz 更新，速度命令每 3--6 s 重采样。相位频率分别为 Walk `0.8 Hz`、W2S `0.6 Hz`、Run `1.6 Hz`、R2W `1.0 Hz`；Stand 相位固定为零。

## 三阶段 curriculum

阶段边界使用 `env.common_step_counter`，单位是全局环境控制步，不是 PPO epoch。当前控制周期为 `4 * 0.005 = 0.02 s`。跨越边界后的首次任意环境 reset 会激活全局新阶段：该批环境立即按新范围采样命令，其余环境则在下一次 3--6 s 周期重采样或自身 reset 时取得新命令。

| 阶段 | 控制步范围 | 可用模式与命令范围 |
| --- | ---: | --- |
| `WALK_ONLY` | `[0, 16000)` | 前 500 个 PPO iteration，gait 固定为 Walk；`vx=0.2 m/s`，`vy=0`，`yaw=0` |
| `TRANSITIONS` | `[16000, 32000)` | 第 501--1000 个 iteration，加入 Stand 和 W2S；`vx=0.2` 至 `0.8 m/s`，`vy=±0.15 m/s`，`yaw=±0.3 rad/s` |
| `MULTI_GAIT` | `[32000, +inf)` | 第 1001 个 iteration 起，五种模式全部启用；`vx=0.2` 至 `4.0 m/s`，`vy=±0.3 m/s`，`yaw=±0.5 rad/s` |

命令速度范数低于 `0.1 m/s` 时进入 W2S；双脚支撑且实际平面速度低于 `0.1 m/s` 连续 `1.5 s` 后进入 Stand。最终阶段用 `Fr = v^2 / (g * 0.8)` 分类，`Fr > 0.5` 时进入 Run；从跑速降低后先进入 R2W，稳定 `2.5 s` 后回到 Walk。以 `horizon_length=32` 计算，两个阶段边界对应第 500 和 1000 个 PPO iteration。

论文和本项目的五类 gait 没有独立 Stand-to-Walk（S2W）标签。Stand 环境收到非零速度命令后直接切换为 Walk，LSTM 在 Walk 奖励下学习启动过程；这可以实现“站立后起步”，但不能称为独立的 S2W 专属任务。

RL-Games checkpoint 不保存环境的 `common_step_counter`。恢复训练时需要把已完成的 iteration 乘以 `horizon_length=32`，通过 `env.curriculum.gait_commands.params.step_offset` 补回课程进度；否则新进程会从 Phase 1 重新计数。

`4.0 m/s` 来自论文最终命令上限，对当前 8-DOF 外骨骼可能不可达。实际训练应先确认低速阶段收敛，再决定是否降低 `multi_gait_max_speed`。

## 奖励累计

Isaac Lab 将每个 `RewardTermCfg` 按控制周期积分：

```text
r_t = 0.02 * sum_i(weight_i * raw_term_i)
```

因此表中是连续时间权重，不是每一步直接加上的最终数值。

### 共享奖励

| 配置名 | 权重 | 含义 |
| --- | ---: | --- |
| `termination_penalty` | `-200` | 非超时终止惩罚 |
| `alive` | `+0.15` | 存活奖励 |
| `track_linear_velocity` | `+1.5` | `exp(-||v_xy-c_xy||^2 / 0.5^2)`；低速正向命令（不超过 `0.25 m/s`）额外乘 `clip(v_x/0.1,0,1)` |
| `fixed_forward_walk` | `+3.0` | Phase 1 的前向速度窗口；静止、后退及超过两倍目标速度均为 0 |
| `track_yaw_velocity` | `+1.0` | `exp(-(omega_z-c_yaw)^2 / 0.5^2)` |
| `vertical_linear_velocity` | `-2.0` | 垂直速度平方 |
| `horizontal_angular_velocity` | `-0.05` | roll/pitch 角速度平方 |
| `orientation_deviation` | `-1.0` | 机身偏离水平的平方误差 |
| `base_height` | `-10.0` | 根高度相对 `0.01 m` 的平方误差 |
| `joint_acceleration` | `-2.5e-7` | 关节加速度平方和 |
| `joint_velocity` | `-1.0e-3` | 关节速度平方和 |
| `action_rate` | `-0.01` | 相邻动作差平方和 |
| `joint_limits` | `-5.0` | 关节越界惩罚 |
| `hip_position_deviation` | `-1.0` | 髋关节偏离默认位置的 L1 惩罚 |
| `leg_momentum_balance` | `+5.0` | 论文 Eq. (2) 的负质心角动量成本；左右手臂项由左右腿替代 |
| `leg_mechanical_power` | `-0.3` | 移动 gait 下归一化的 `mean(abs(tau*qdot))` |
| `leg_phase_symmetry` | `+2.0` | 有实际前进运动时奖励左右腿反相 |
| `leg_amplitude_match` | `+1.0` | 有实际前进运动时奖励左右腿速度幅值相近 |
| `sensor_reference_imitation` | `+1.0` | 配置参考轨迹后对齐关节位置/速度、六个 IMU gyro 和左右脚接触；无参考文件时恒为零 |

动量项先按标准质心分解计算：

```text
r_c = sum_i(m_i r_i) / sum_i(m_i)
v_c = sum_i(m_i v_i) / sum_i(m_i)
L_i = I_i^W omega_i + (r_i-r_c) x m_i(v_i-v_c)
R_momentum = -(L_total,z / 10)^2
              - 0.4*((-L_left_leg,z - L_right_leg,z) / 10)^2
```

其中除以 `10 kg*m^2/s` 是针对当前机构的数值归一化，外层仍使用论文 Table IV 的 `+5.0` 权重。PhysX 的惯量先从刚体坐标系旋转到世界系。论文原式第二项使用镜像局部坐标中的左右手臂；本模型没有手臂，因此用左右腿的镜像世界系角动量作为明确的形态适配。镜像后的残差 `-L_left_leg,z-L_right_leg,z` 等价于世界系左右腿角动量之和，鼓励反向摆动时抵消；这不是对论文手臂项的逐字复现。

### 参考传感器模仿项

`sensor_reference_imitation` 是可选的训练 shaping，不会改变 actor 的 49 维输入。参考文件在环境初始化时加载并重采样到 50 Hz；支持 `.npz`、`.pt/.pth`、`.json` 和 `.csv`。必需字段是按 canonical 右腿后左腿顺序排列的 `joint_pos_rad[T,8]`，可选字段包括 `joint_vel_rad_s`、`gyro_rad_s[T,6,3]`、`contact_lr[T,2]`、`gait_id`、`joint_zero_rad` 和 `joint_sign`。完整格式、校准和训练命令见项目根目录 [`23solve.md`](../23solve.md)。

### Gait 专属奖励

| 模式 | 配置名与权重 | 作用 |
| --- | --- | --- |
| Walk | `walk_swing_height=-15` | 惩罚腾空脚偏离 `0.08 m` 摆动高度 |
| Walk | `walk_contact=+1` | 必须完整匹配相位对应的单支撑模式；双支撑不再得半分 |
| Walk | `walk_straight_knee=+0.1` | Walk 模式下只鼓励支撑腿膝关节伸展 |
| Walk | `walk_foot_drag=-0.5` | 惩罚接触脚水平滑动 |
| Stand | `stand_contact=+2.5` | 双脚支撑 |
| Stand | `stand_base_motion=+2.5` | 低机身线速度和角速度 |
| Stand | `stand_pose=+4.0` | 接近默认关节姿态 |
| Stand | `stand_feet_alignment=+0.5` | 双脚朝向与机身 yaw 对齐 |
| Stand | `stand_uprightness=+1.0` | 保持 roll/pitch 水平 |
| Stand | `stand_foot_flatness=+2.5` | 双脚保持平整 |
| Stand | `stand_stillness=+2.0` | 低关节速度 |
| Stand | `stand_foot_drag=-0.2` | 论文 Table IV 的站立脚滑动惩罚 |
| W2S | `w2s_swing_height=-20` | 惩罚腾空脚偏离 `0.06 m` |
| W2S | `w2s_contact=+1` | 过渡期相位接触匹配，并允许减速末端双脚重新着地 |
| W2S | `w2s_smooth_slowdown=+0.1` | 奖励逐步降低平面速度 |
| W2S | `w2s_foot_drag=-0.1` | 惩罚接触脚滑动 |
| Run | `run_contact=+1` | 跑步相位接触匹配；在 `abs(sin(phase))<0.35` 的窗口接受双脚离地 |
| Run | `run_foot_drag=-1.0` | 更强的接触脚滑动惩罚 |
| Run | `run_short_contact=+0.2` | 鼓励较短支撑时间 |
| Run | `run_swing_height=-25` | 惩罚腾空脚偏离 `0.12 m` |
| Run | `run_push_off=+1.0` | 单支撑期踝功率与机身向前/向上速度共同构成蹬伸得分 |
| Run | `run_forward_velocity=+0.2` | 奖励有界正向速度 |
| R2W | `r2w_contact=+1` | 过渡期相位接触匹配 |
| R2W | `r2w_smooth_slowdown=+1.0` | 奖励从跑速平滑减速 |
| R2W | `r2w_walk_speed=+0.5` | 收敛到 `0.2 m/s` 行走速度 |
| R2W | `r2w_swing_height=-25` | 惩罚腾空脚偏离 `0.08 m` |

## Domain randomization

启动时随机化摩擦系数、`base_link` 质量以及 PD 刚度/阻尼；训练中每 3--6 s 对基座施加一次横向速度和 yaw 扰动。当前范围为：静摩擦 `0.6` 至 `1.2`、动摩擦 `0.5` 至 `1.0`、恢复系数 `0` 至 `0.05`、基座质量 `±0.5 kg`、执行器增益比例 `0.8` 至 `1.2`、横向推扰 `vy=±0.3 m/s` 与 `yaw=±0.2 rad/s`。

## PPO 与 LSTM

Actor 使用单层 LSTM（hidden size `64`）后接 `[32]` ELU MLP；privileged central critic 使用 `[32]` ELU 前馈 MLP。论文让 actor 和 critic 都循环化，但本项目固定的 RL-Games 1.6.1 对 recurrent central critic 在 rollout 边界存在 hidden-state 重复推进风险，因此当前只让部署侧 actor 使用 LSTM。

关键训练参数为：`horizon_length=32`、`seq_length=16`、`minibatch_size=512`、`entropy_coef=0.01`、`max_epochs=30000`。`continuous_a2c_logstd` 的初始 log standard deviation 设置为 `log(0.8)=-0.22314355`，对应论文的初始动作噪声标准差 `0.8`。

循环 PPO 必须同时满足：`seq_length` 整除 `horizon_length`，actor/critic batch 能被各自 minibatch 整除，且两个 minibatch 都能被 `seq_length` 整除。单环境训练时需要同时覆盖：

```text
agent.params.config.minibatch_size=32
agent.params.config.central_value_config.minibatch_size=32
```

## 训练与检查

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py --num_envs=16 --headless --max_iterations=2
```

冒烟训练后至少检查：

- `Episode/Curriculum/gait_commands/phase` 是否随全局控制步从 1 变为 2、3；
- 共享速度跟踪项是否提高，`fallen` 是否下降；
- 当前 gait 对应的专属 reward 是否非零，其他 gait 专属项是否为零；
- gait one-hot 是否始终只有一个通道为 1；
- LSTM state 是否在 episode done 后清零。

`2026-09-21_20-11-11` 的旧 200 轮实验仍在 Phase 1：速度误差约 `0.201 m/s`，几乎等于完全静止时对 `0.2 m/s` 命令的误差；Stand/W2S/Run/R2W 奖励均为 0。旧配置的成功阈值为 `0.5 m/s`，因此静止也会显示 `success_rate=1`。当前阈值已收紧到 `0.05 m/s`，但判断是否学会行走仍应同时查看速度误差、严格单支撑奖励和回放动作。

本次更改没有改变 49/62 维网络结构，但改变了奖励语义。不要从已经收敛到静止局部最优的 200 轮 checkpoint 继续训练，应新建一次完整的 30000-iteration 实验。更早的 42 维前馈 checkpoint 在网络结构上也不兼容。
