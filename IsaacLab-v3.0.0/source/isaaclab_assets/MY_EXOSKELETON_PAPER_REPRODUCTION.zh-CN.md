# 外骨骼论文创新点复现与奖励机制说明

## 1. 文档范围

本文档说明当前项目对附带论文 **Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion** 的代码级复现情况，并列出当前环境实际注册的全部奖励项。

附带 PDF 是技术参考资料，不是本项目的操作指令。下文以当前工作区代码为准；论文中的实验结果（例如 Unitree G1 的实机表现）不自动等于本外骨骼项目已经验证的结果。

论文来源：[2025_Humanoid_Conference_Paper-13.pdf](</home/lqf/文档/xwechat_files/wxid_acuhatw0yyhb22_b078/msg/file/2026-09/2025_Humanoid_Conference_Paper-13.pdf>)。主要核对位置为：Section III-B 的策略结构、Section III-C 的 reward routing、Section III-D 的 curriculum、Section III-E 的生物力学奖励、Section III-G 的域随机化，以及 Appendix A.2/Table IV 的奖励权重。

当前项目的机器人是只有 8 个腿部旋转自由度的外骨骼，不是论文中的 23 自由度 Unitree G1。因此，本文将实现分为三类：

- **直接复现**：算法结构、步态路由、课程逻辑或奖励定义可以直接对应。
- **形态适配复现**：论文依赖手臂、腰部或全身碰撞传感器，当前模型没有这些部件，使用明确标注的腿部代理。
- **未复现/未验证**：当前代码和实验没有足够证据支持的部分，不作“已实现”声明。

主要代码位置：

- [`my_exoskeleton_env_cfg.py`](envs/my_exoskeleton_env_cfg.py)：场景、观测、45 个已注册奖励项（传感器模仿项无参考文件时为零）、事件和课程配置。
- [`my_exoskeleton_flat.py`](tasks/my_exoskeleton_flat.py)：命令状态机、观测构造和奖励函数实现。
- [`gait.py`](tasks/gait.py)：五种 gait ID、Froude 判据和课程边界。
- [`my_exoskeleton_ppo.yaml`](configs/my_exoskeleton_ppo.yaml)：PPO、LSTM 和 privileged critic 配置。
- [`MY_EXOSKELETON_REWARDS.zh-CN.md`](../../scripts/MY_EXOSKELETON_REWARDS.zh-CN.md)：奖励公式的扩展说明。

## 2. 论文创新点与当前复现状态

| 论文创新点 | 当前项目状态 | 复现说明 |
| --- | --- | --- |
| 一个策略统一执行 Stand、Walk、Run 及过渡 | 已实现，形态适配 | 一个 49 维 actor、8 维动作策略覆盖 `Stand/Walk/W2S/Run/R2W`；没有为每种 gait 建立独立策略。 |
| gait ID 条件输入与奖励路由 | 已直接实现 | 五维 one-hot gait ID 进入 actor/critic；每个 gait 专属奖励按当前 gait mask 路由，非当前 gait 的项严格为零。 |
| 无 MoCap 的 reference-free 学习 | 已实现 | 目标来自速度命令、接触、姿态和生物力学约束，不读取动作捕捉轨迹。 |
| 生物力学奖励 shaping | 已实现并做腿部适配 | 保留总质心角动量、直膝、接触相位、摆脚高度、拖脚、推蹬和过渡平滑等机制；手臂相关项用腿部代理替代。 |
| 多阶段 gait/command curriculum | 已实现 | 当前每 500 个 PPO iteration 开放一阶段；第 500 轮进入 Transitions，第 1000 轮进入 Multi-gait。 |
| 循环策略处理部分可观测性 | 部分直接复现 | actor 使用单层 64 单元 LSTM；当前 RL-Games 配置的 privileged central critic 使用前馈 MLP，未强行配置 critic LSTM。 |
| 非对称 actor-critic 与 privileged state | 已实现，输入不同 | actor 49D，critic 62D；critic 额外看到脚接触、接触力、根高度和关节加速度。 |
| 域随机化和外部扰动 | 已实现 | 随机化摩擦、基座质量、执行器 PD 增益，并周期性施加横向/偏航扰动。 |
| Unitree G1 实机迁移 | 未验证 | 当前项目只对 Isaac Lab 仿真训练链路做验证，不能引用论文的实机结论。 |

## 3. 当前系统的步态和训练策略

### 3.1 Gait ID

策略接收固定顺序的五维 one-hot：

| ID | 模式 | 触发条件/用途 |
| ---: | --- | --- |
| `0` | Stand | 低速命令、双脚支撑并且实际速度低于 `0.1 m/s` 持续 `1.5 s`。 |
| `1` | Walk | 常规非零速度命令；也是第一阶段唯一启用的 gait。 |
| `2` | Walk-to-Stand (W2S) | 命令速度范数低于 `0.1 m/s` 后的减速与接触重塑阶段。 |
| `3` | Run | Multi-gait 阶段使用 `Fr = v^2/(g*0.8) > 0.5` 判定。 |
| `4` | Run-to-Walk (R2W) | 从 Run 降速后，稳定 `2.5 s` 再回到 Walk。 |

状态机以 `10 Hz` 更新，速度命令每 `3--6 s` 重采样。论文和本项目都没有独立的 Stand-to-Walk gait ID；Stand 收到非零命令后直接转到 Walk，起步过程由循环策略学习。

### 3.2 三阶段课程

当前 PPO `horizon_length=32`，所以 500 个 iteration 等于 `16,000` 个控制步：

| PPO iteration | 控制步 | 阶段 | 命令和 gait |
| ---: | ---: | --- | --- |
| `0--499` | `0--15,999` | `WALK_ONLY` | 固定 `vx=0.2 m/s`，只启用 Walk。 |
| `500--999` | `16,000--31,999` | `TRANSITIONS` | 速度上限 `0.8 m/s`，加入 Stand 和 W2S。 |
| `>=1000` | `>=32,000` | `MULTI_GAIT` | 速度上限 `4.0 m/s`，启用五种 gait。 |

课程边界以环境全局控制步计数为准，不是单个物理步，也不是 checkpoint 文件中的 epoch 编号。RL-Games checkpoint 不保存该计数；恢复时需要设置：

```text
env.curriculum.gait_commands.params.step_offset = 已完成的 iteration 数 * 32
```

## 4. 奖励计算和路由规则

### 4.1 累计方式

控制周期为 `dt=0.02 s`。Isaac Lab 对每个奖励项先计算 raw value，再乘配置权重并按控制周期积分：

```text
r_t = 0.02 * sum_i(weight_i * raw_i)
```

因此下表的 weight 是配置权重，不是每个物理子步直接加到回报中的数值。

### 4.2 Gait mask

对 gait 专属项使用：

```text
M_g(t) = 1, gait_id(t) == g
       = 0, otherwise
r_g(t) = M_g(t) * raw_g(t)
```

共享项（速度跟踪、姿态、存活和正则项）通常对所有 gait 生效。动量、机械功率、双腿反相和幅值匹配在实现中额外使用 locomotion mask，在静态 Stand 时为零；第一阶段的 `fixed_forward_walk` 只在 Walk/固定低速命令下有效。接触判定阈值为 `5 N`。

路由概览：

| 奖励组 | Stand (0) | Walk (1) | W2S (2) | Run (3) | R2W (4) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 常规共享项 | 1 | 1 | 1 | 1 | 1 |
| `fixed_forward_walk` | 0 | 仅固定低速命令 | 0 | 0 | 0 |
| 动量/功率/反相/幅值 | 0 | 1 | 1 | 1 | 1 |
| Walk 专属组 | 0 | 1 | 0 | 0 | 0 |
| Stand 专属组 | 1 | 0 | 0 | 0 | 0 |
| W2S 专属组 | 0 | 0 | 1 | 0 | 0 |
| Run 专属组 | 0 | 0 | 0 | 1 | 0 |
| R2W 专属组 | 0 | 0 | 0 | 0 | 1 |

## 5. 全部奖励机制

当前 Reward Manager 注册 **45 项**：共享 19 项（含可选的传感器模仿项），Walk 4 项，Stand 8 项，W2S 4 项，Run 6 项，R2W 4 项。未配置参考文件时，传感器模仿项返回零。

### 5.1 共享奖励和正则项（19 项，含一个可选项）

| 配置名 | weight | raw 机制 | 论文对应/备注 |
| --- | ---: | --- | --- |
| `termination_penalty` | `-200` | 非 timeout 的 episode 终止惩罚 | 论文 collision penalty 的安全代理；当前没有全身碰撞传感器。 |
| `alive` | `+0.15` | 存活奖励 | 对应论文 alive bonus。 |
| `track_linear_velocity` | `+1.5` | `exp(-||v_xy-c_xy||^2 / 0.5^2)`；第一阶段低速命令还乘 `clip(vx/0.1,0,1)` | 速度任务奖励；低速 activity gate 用于消除静止局部最优。 |
| `fixed_forward_walk` | `+3.0` | 以 `0.2 m/s` 为中心的前向速度窗口，并惩罚横向速度和 yaw 误差 | 项目新增的第一阶段探索辅助项，静止时为零。 |
| `track_yaw_velocity` | `+1.0` | `exp(-(omega_z-c_omega)^2 / 0.5^2)` | 对应 angular velocity tracking。 |
| `vertical_linear_velocity` | `-2.0` | 根部垂直速度平方 | 论文 Table IV，直接对应。 |
| `horizontal_angular_velocity` | `-0.05` | roll/pitch 角速度平方 | 论文 Table IV，直接对应。 |
| `orientation_deviation` | `-1.0` | 根部偏离水平的平方误差 | 论文 orientation deviation。 |
| `base_height` | `-10.0` | 根高相对 `0.01 m` 的平方误差 | 论文 base height。 |
| `joint_acceleration` | `-2.5e-7` | 关节加速度平方和 | 论文 DOF acceleration。 |
| `joint_velocity` | `-1e-3` | 关节速度平方和 | 论文 DOF velocity。 |
| `action_rate` | `-0.01` | 相邻归一化动作差平方和 | 论文 action-rate 正则。 |
| `joint_limits` | `-5.0` | 关节越限惩罚 | 论文 DOF-limit 正则。 |
| `hip_position_deviation` | `-1.0` | 两侧髋关节偏离默认位置的 L1 | 论文 hip position deviation。 |
| `leg_momentum_balance` | `+5.0` | 见 5.2 的质心角动量公式 | 论文 arm-leg momentum balance 的腿部形态代理。 |
| `leg_mechanical_power` | `-0.3` | `clip(mean(abs(tau*qdot))/50,0,10)` | 论文 arm-swing energy 的腿部机械功率代理。 |
| `leg_phase_symmetry` | `+2.0` | 奖励左右腿速度反相 `qdot_R + qdot_L ~= 0` | 论文 arm swing symmetry 的腿部代理；仅实际前进时开放。 |
| `leg_amplitude_match` | `+1.0` | 奖励左右腿速度幅值接近 | 论文 arm-leg amplitude match 的腿部代理；仅实际前进时开放。 |
| `sensor_reference_imitation` | `+1.0` | 对齐参考关节位置、速度、六个 IMU gyro 和左右脚接触；缺少的可选通道会被移除并重新归一化 | 项目新增的可选数据指导项；没有 `reference_data_path` 时恒为零。 |

其中动量项使用真实刚体数据。令 `L_total,z` 为所有刚体的总质心角动量 z 分量，`L_left,z`、`L_right,z` 为显式分组的左右腿角动量，则代码实现为：

```text
r_momentum = -(L_total,z / 10)^2
             - 0.4 * ((-L_left,z - L_right,z) / 10)^2
```

每个刚体的轨道项使用 `(r_i-r_c) x m_i(v_i-v_c)`，自旋项使用世界系惯量乘世界系角速度。惯量会先由刚体坐标系旋转到世界系。除以 `10 kg*m^2/s` 是当前外骨骼的数值归一化，不是论文原式中的新物理假设。

### 5.2 Walk 专属奖励（4 项）

| 配置名 | weight | raw 机制 | 路由 |
| --- | ---: | --- | --- |
| `walk_swing_height` | `-15` | 腾空脚高度与 `0.08 m` 的平方误差 | 仅 `gait_id=Walk`。 |
| `walk_contact` | `+1` | 相位预测的左右单支撑接触完全匹配 | 仅 Walk；双脚同时接触不再获得半分。 |
| `walk_straight_knee` | `+0.1` | 只对当前支撑腿的膝关节伸展给分 | 仅 Walk，避免用摆动腿作弊。 |
| `walk_foot_drag` | `-0.5` | 接触脚水平速度范数 | 仅 Walk，惩罚拖脚。 |

### 5.3 Stand 专属奖励（8 项）

| 配置名 | weight | raw 机制 | 路由 |
| --- | ---: | --- | --- |
| `stand_contact` | `+2.5` | 左右脚同时接触地面 | 仅 Stand。 |
| `stand_base_motion` | `+2.5` | 根部线速度和角速度低时的指数分数 | 仅 Stand。 |
| `stand_pose` | `+4.0` | 关节接近默认站立姿态 | 仅 Stand。 |
| `stand_feet_alignment` | `+0.5` | 双脚 yaw 与根部方向对齐 | 仅 Stand。 |
| `stand_uprightness` | `+1.0` | roll/pitch 接近零 | 仅 Stand。 |
| `stand_foot_flatness` | `+2.5` | 双脚 roll/pitch 保持平整，忽略 yaw | 仅 Stand。 |
| `stand_stillness` | `+2.0` | 关节速度低时的指数分数 | 仅 Stand。 |
| `stand_foot_drag` | `-0.2` | 站立时接触脚水平滑动 | 仅 Stand。 |

### 5.4 Walk-to-Stand 专属奖励（4 项）

| 配置名 | weight | raw 机制 | 路由 |
| --- | ---: | --- | --- |
| `w2s_swing_height` | `-20` | 腾空脚高度与 `0.06 m` 的误差 | 仅 W2S。 |
| `w2s_contact` | `+1` | 相位接触匹配；双支撑 credit 在 `1.5 s` 内渐增 | 仅 W2S。 |
| `w2s_smooth_slowdown` | `+0.1` | `clip((previous_speed-current_speed)/0.2,0,1)` | 仅 W2S，奖励减速而非突然停住。 |
| `w2s_foot_drag` | `-0.1` | 接触脚水平滑动 | 仅 W2S。 |

### 5.5 Run 专属奖励（6 项）

| 配置名 | weight | raw 机制 | 路由 |
| --- | ---: | --- | --- |
| `run_contact` | `+1` | 相位接触匹配；`abs(sin(phase))<0.35` 时允许双脚离地 | 仅 Run。 |
| `run_foot_drag` | `-1.0` | 接触脚水平滑动 | 仅 Run，惩罚更强。 |
| `run_short_contact` | `+0.2` | `exp(-contact_time/0.25)`，鼓励短支撑 | 仅 Run。 |
| `run_swing_height` | `-25` | 腾空脚高度与 `0.12 m` 的误差 | 仅 Run。 |
| `run_push_off` | `+1.0` | 单支撑期踝功率与正向/向上根部运动的乘积 | 仅 Run，模拟 push-off。 |
| `run_forward_velocity` | `+0.2` | 有界正向根部速度 `clip(vx/4.0,0,1)` | 仅 Run。 |

### 5.6 Run-to-Walk 专属奖励（4 项）

| 配置名 | weight | raw 机制 | 路由 |
| --- | ---: | --- | --- |
| `r2w_contact` | `+1` | 过渡期相位接触匹配 | 仅 R2W。 |
| `r2w_smooth_slowdown` | `+1.0` | 奖励速度逐步下降 | 仅 R2W。 |
| `r2w_walk_speed` | `+0.5` | 根部前向速度接近 `0.2 m/s` | 仅 R2W。 |
| `r2w_swing_height` | `-25` | 腾空脚高度与 `0.08 m` 的误差 | 仅 R2W。 |

## 6. 论文奖励与当前奖励的对应关系

| 论文机制 | 当前实现 | 结论 |
| --- | --- | --- |
| `R_momentum = -L_total,z^2 - 0.4(L_left_arm,z-L_right_arm,z)^2` | 真实 PhysX 质心角动量 + 左右腿镜像残差 | **等价目标，非逐字复现**；没有手臂，只能做腿部形态适配。 |
| Arm energy | `leg_mechanical_power`，`abs(tau*qdot)` | **形态代理**；当前权重为负惩罚，避免高机械功率。 |
| Arm symmetry | `leg_phase_symmetry` | **形态代理**；鼓励左右腿反相。 |
| Arm-leg amplitude match | `leg_amplitude_match` | **形态代理**；匹配左右腿摆动幅值。 |
| Straight knee | `walk_straight_knee` | 直接保留，但只在 Walk 的支撑腿上生效。 |
| Contact/foot clearance/foot drag | 各 gait 的接触、摆脚高度和拖脚项 | 直接复现目标，阈值和目标高度按外骨骼足端调整。 |
| Push-off/short contact | `run_push_off`、`run_short_contact` | 直接复现目标，使用踝功率和接触传感器代理。 |
| Standing stability | Stand 的双支撑、姿态、平足、静止项 | 直接复现目标，适配为腿部可观测量。 |
| Collision penalty | `termination_penalty=-200` | **安全代理**；没有论文所需的全身碰撞类别/传感器。 |

论文 Table IV 中的调节权重（如 `-2.0` 垂直速度、`-0.05` 水平角速度、`-10` 根高、`-2.5e-7` 加速度、`-1e-3` 速度、`-0.01` action rate、`-5` 关节限位和 `+0.15` alive）已在当前配置中保留。Arm/elbow/waist/shoulder 项无法在 8 自由度外骨骼上直接计算，未伪造同名信号。

## 7. 结构和传感器差异

| 项目 | 论文 | 当前外骨骼项目 |
| --- | --- | --- |
| 机器人形态 | Unitree G1，全身含手臂、腰和肩肘 | 8 个腿部旋转关节 |
| 动作 | 论文为 23 维关节位置偏移 | 8 维相对默认姿态的位置偏移，实际目标为“默认关节角 + 动作 × 缩放”，按关节使用 `0.45/0.65/0.95/0.50 rad`（左右腿同构） |
| Actor 输入 | 本体状态、命令、相位和 gait ID | 49D，含脚接触/压力统计、相位和 5D gait one-hot |
| Critic 输入 | actor 观测加多种 privileged state | 62D，额外含双脚接触标志、归一化力、根高和关节加速度 |
| 循环结构 | 论文 actor 和 critic 都使用 LSTM | actor 使用 LSTM；central critic 使用 `[32]` ELU 前馈 MLP，以兼容当前 RL-Games rollout |
| 接触观测 | 论文包含更丰富的接触/扰动信息 | 两个足端 ContactSensor，接触阈值 `5 N`，用力的均值/标准差作代理 |
| 实机验证 | G1 站立、行走和 W2S | 当前尚未做外骨骼实机验证 |

论文中的 `min torso angular velocity`、waist pitch/roll/yaw、torso yaw smoothness、shoulder roll control 和 elbow phase tracking 也没有在当前 Reward Manager 中启用；外骨骼没有相应的躯干、肩或肘自由度。论文表格中的 torque regulation 同样没有作为独立 `RewardTermCfg` 注册，不能把未注册的辅助函数当作已启用奖励。

## 8. 复现与验证方法

### 8.1 静态测试

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
uv run pytest -q \
  source/isaaclab_assets/test/test_my_exoskeleton_env_cfg.py \
  source/isaaclab_assets/test/test_my_exoskeleton_gait.py \
  source/isaaclab_assets/test/test_play_my_exoskeleton.py
```

当前基线为 `92 passed`（Kit-less 回归，完整 Kit/PhysX 配置测试另行运行）。测试覆盖 gait ID、Froude 阈值、500/1000 轮课程边界、gait mask、低速 activity gate、接触模式、W2S 双支撑、Run 腾空、质心惯量坐标变换、传感器参考数据加载、多模态数据集、固定相机、事件通道有效性和非循环目标序列。

### 8.2 Isaac Sim 冒烟训练

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --num_envs=16 --headless --max_iterations=2
```

应确认无 traceback，并看到 actor `49` 维、critic `62` 维、动作 `8` 维以及 45 个已注册 reward terms。

### 8.3 课程和行为验收

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --num_envs=256 --headless --max_iterations=1500
```

建议监控：

- `Episode/Curriculum/gait_commands/phase` 是否在约第 500/1000 轮变为 2/3；
- `error_vel_xy`、`fallen`、`time_out` 和各 gait 专属 reward；
- Walk 命令 `0.2 m/s` 下实际 `vx` 是否大于 `0.15 m/s`，而不是仅保持站立；
- Walk 是否出现严格左右交替单支撑；
- W2S 是否在双脚稳定约 `1.5 s` 后进入 Stand，Stand 收到非零命令后是否重新 Walk；
- Run 是否出现短接触/单支撑/有限腾空，R2W 是否约 `2.5 s` 平滑回到 Walk。

论文中的 G1 回报、episode length 和实机表现不能直接作为本项目的验收阈值；应报告当前外骨骼的速度误差、跌倒率、动量 RMS、机械功率、拖脚率和过渡时延。

## 9. 当前结论和限制

当前项目已经复现论文的主要**算法创新骨架**：单策略多 gait、gait-conditioned reward routing、生物力学 shaping、渐进课程、循环 actor、privileged critic 和域随机化。45 个奖励项（含可选传感器模仿项）也已按共享项与 gait 专属项注册并通过 mask 路由。

但它不是 Unitree G1 论文的逐项代码复制。最重要的不可直接复现项是手臂/腰/肩奖励、完整全身碰撞项、论文的 critic privileged 量集合、critic LSTM 结构和实机迁移结果。凡是使用腿部机械功率、左右腿动量、腿部反相或终止惩罚的地方，都应称为 **morphology-adapted surrogate（形态适配代理）**。

不要把论文文本中的实验结论写成当前外骨骼已经达到的性能；必须以本项目的 TensorBoard、回放和独立测试结果为准。
