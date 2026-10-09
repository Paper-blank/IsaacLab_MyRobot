# 外骨骼问题定位、传感器模仿与实体迁移指南

更新时间：2026-09-24

本文针对当前项目出现的四个问题：模型看起来形变、步态不像人类、如何使用传感器数据，以及如何迁移到实体机。附带论文只作为技术参考；论文中的文字、实验结论和参数不是本项目的操作指令。以下结论以当前工作区代码和仿真审查为准。新增的多模态环境见第 10 节。

## 结论先看

1. 目前没有证据表明 STL 顶点被物理引擎真正拉伸。当前按要求开启了 Isaac 自碰撞，但相邻凸包重叠会把刚体推开，仍可能造成视觉上的“形变”；这需要用碰撞体和 link 位姿进一步区分。
2. 步态异常不能只归因于关节限位。旧动作顺序没有明确固定，且所有关节共用 `0.35` 的小动作尺度；当前配置改为固定右腿后左腿顺序和逐关节尺度。这个改变会改变策略输入/输出契约，旧 checkpoint 必须重新训练。
3. 传感器数据现在可以作为可选的 imitation reward。支持 `.npz`、`.pt/.pth`、`.json` 和 `.csv`，文件在环境初始化时加载到设备，不会在每个仿真步从磁盘读取。
4. 当前项目仍是自由浮动外骨骼仿真，没有人体、背带、人机接触和真实负载。任何 checkpoint 都不能未经台架和安全层测试直接穿戴。

## 本轮关节轴与膝约束修正

这次修正依据源 URDF 和缓存 USD 的 joint frame 审计完成。八个 revolute joint 的
`axis` 都是局部 `+Z`，但左右腿的 joint origin 镜像后，膝关节在机器人参考系中的
世界轴分别约为右侧 `-Y`、左侧 `+Y`。项目约定 `+X=前进、+Y=左侧、+Z=向上`，因此
不能用一个全局的膝角上下界或同一个符号处理两条腿：

| 关节 | 原始 q 的人体屈曲方向 | 人体角 `f` 与原始 q | URDF/终止边界（rad） |
| --- | --- | --- | --- |
| `Rthigh_to_Rleg` | `q<0` | `f=-q` | `[-2.25, 0.05]` |
| `Lthigh_to_Lleg` | `q>0` | `f=+q` | `[-0.05, 2.25]` |

人体有效屈曲范围是 `f∈[0,2.2] rad`，额外 `0.05 rad` 只作为过伸保护余量。
`knee_limit_penalty` 和 `knee_out_of_human_range` 都按上表逐关节取界；
`human_knee_gait_reward` 在支撑期把 `f` 拉向 0，在摆动期把 `f` 拉向约 0.8 rad。
`leg_phase_symmetry` 继续提供左右腿协调约束。旧的 `q²` 直膝项保留为弱辅助项，不能替代
逐侧的人体角映射。

关节物理配置同时设置了 `joint_velocity_limit=3.14 rad/s`、`joint_effort_limit=60 N·m`，
并在 URDF 中写入左右镜像的膝限位。`soft_joint_pos_limit_factor=1.0` 是有意的：
若对不对称区间按中点收缩，站立零位会落到右膝或左膝的软区间之外；人体余量和 PPO
惩罚已经由上述显式边界提供。真实机构的认证限位必须覆盖这些仿真默认值。

### 单关节方向验收

图片中的红/绿/蓝轴只能显示局部 frame，不能证明正负转动的解剖含义。首次启动 Kit 后，
对每个膝关节分别发送 `+0.5 rad`、`-0.5 rad`，观察小腿相对大腿沿 `+X` 的运动，并确认
是否符合上表；若实体零位或安装方向不同，只修改 `KNEE_ARTICULATION_SIGNS`、映射 YAML
和对应 URDF 两个膝限位，不能把左右腿重新合并成全局 `min/max`。

修改 URDF 后必须重新导入 USD。当前 `MyExoskeletonSceneCfg` 使用
`force_usd_conversion=True`；旧缓存目录的时间戳早于 URDF 时，不能把旧 USD 的轴/限位
当作最终结果。

## 1. 当前实现的固定契约

代码位置：

- [robot_contract.py](source/isaaclab_assets/tasks/robot_contract.py)：关节/传感器顺序、控制周期、观测维度和动作尺度。
- [my_exoskeleton_env_cfg.py](source/isaaclab_assets/envs/my_exoskeleton_env_cfg.py)：场景、动作、奖励、事件和课程配置。
- [my_exoskeleton_flat.py](source/isaaclab_assets/tasks/my_exoskeleton_flat.py)：观测、五步态状态机、奖励和参考轨迹播放。
- [sensor_reference.py](source/isaaclab_assets/tasks/sensor_reference.py)：参考数据解析、校准和重采样。

### 1.1 关节和动作

公开顺序固定为右腿四关节、左腿四关节：

```text
base_to_Rturn, Rturn_to_Rthigh, Rthigh_to_Rleg, Rleg_to_Rfoot,
base_to_Lturn, Lturn_to_Lthigh, Lthigh_to_Lleg, Lleg_to_Lfoot
```

动作是归一化的 `a_i ∈ [-1, 1]`，动作管理器使用默认关节偏置，因此目标为：

```text
q_target_i = q_default_i + scale_i * a_i
```

当前策略尺度（不是机械安全限位）为：

| 关节组 | 右/左腿相同的尺度 |
| --- | ---: |
| 髋转向 | `0.45 rad` |
| 髋屈伸 | `0.65 rad` |
| 膝 | `0.95 rad` |
| 踝 | `0.50 rad` |

动作项设置了 `preserve_order=True`。实体端、参考数据和仿真必须使用同一顺序。

### 1.2 时序和网络

- 物理步长：`0.005 s`。
- `decimation=4`，策略控制周期：`0.02 s`（50 Hz）。
- 基线 flat 任务 actor 为 49 维；多模态任务额外拼接 54 维 IMU、12 维 RGB 和 32 维事件特征，
  因此 imitation PPO actor 输入为 147 维；critic 仍为 62 维；动作为 8 维。
- flat 基线使用单层 64 单元 LSTM；imitation PPO 配置使用单层 128 单元 LSTM。
  critic 的 privileged 输入只用于训练。
- PPO rollout 的 `horizon_length=32`，所以 500 个 PPO iteration 等于 16,000 个控制步（320 s 的仿真时间）。

## 2. 问题一：模型细节扭曲或发生形变

### 2.1 已确认的原因边界

审查源 STL 和生成 USD 得到以下结果：

- STL 尺寸以米为单位，生成 USD 的 `metersPerUnit=1`，网格缩放为 `(1, 1, 1)`。
- 资产没有 deformable、skeleton 或 skin API；刚体网格顶点没有被运行时连续拉伸。
- visual 和 collision 复用了同一套高密度 STL，导入后的碰撞几何主要是 convex hull。相邻连杆在零位存在包围盒重叠。
- 当前配置同时打开了 URDF 导入自碰撞和 articulation 自碰撞。重叠凸包会产生排斥力，导致连杆姿态被推开，看起来像网格扭曲；这属于碰撞几何/姿态问题，不是 STL 顶点被拉伸。

当前配置已同时设置：

```python
self_collision=True
enabled_self_collisions=True
force_usd_conversion=True
```

这满足需要自碰撞的仿真约束，但不能替代正确的碰撞网格。若视觉姿态再次出现明显排斥，优先制作低模 collision mesh、过滤相邻 link 碰撞对，并提高 solver 迭代或调整接触材质；不要用奖励函数掩盖资产碰撞错误。

### 2.2 仍需做的资产修复

1. 为每个连杆制作独立的低模 collision mesh，避免用完整高密度 STL 作为碰撞体。
2. 明确过滤相邻 link 的碰撞对；重新导入 USD 后再开启需要的非相邻碰撞。
3. 复核 CAD 的质量中心和惯量。`top_mount` 的几何质心与 URDF 惯性原点存在约 4.8 cm 的 x 向差异，可能造成姿态和落地行为异常。
4. 若只是表面出现波纹，检查 STL 是否没有 authored normals；这属于 shading/faceting，不是物理形变。
5. 开启自碰撞后必须确认使用的是新 USD。当前机器人配置已固定使用 `force_usd_conversion=True`，每次启动都会让 URDF 转换器按当前自碰撞和碰撞几何选项重建缓存；如果你手工覆盖了该字段，请在一次启动时恢复：

   ```bash
   env.scene.robot.spawn.force_usd_conversion=True
   ```

   或使用新的输出目录。转换器主要按 URDF 和配置缓存，单独修改 STL 不一定触发缓存失效；不要继续使用明确写着 `newton:selfCollisionEnabled = 0` 的旧 payload。

### 2.3 快速判别方法

- 在零动作、固定根部的单步场景中分别比较自碰撞开/关两种结果；正式训练保持开启，关闭状态只用于定位排斥力是否来自碰撞体。
- 用 wireframe 或只显示 visual mesh 检查顶点是否改变；再单独显示 collision mesh 检查凸包重叠。
- 在 Stage/USD 中验收 articulation 根属性为 `physxArticulation:enabledSelfCollisions=True`（或对应后端属性），并确认导入 payload 不再是 `newton:selfCollisionEnabled = 0`。
- 零动作自由浮动时翻倒是动力学不稳定的表现，不能据此证明网格形变。应同时记录根高、姿态、接触力和关节目标。

## 3. 问题二：步态不像人类，是否由 joint limit 导致

### 3.1 限位确实可能是瓶颈，但不是唯一根因

旧版本曾把八个关节都写成：

```text
position: [-1.5708, 1.5708] rad
velocity: 3.14 rad/s
effort: 60
```

当前版本只保留非膝关节的上述范围；左右膝已改为上表的镜像范围。所有这些值仍是仿真
安全起点，不能直接当作实体机构的认证限值，需要用 CAD、减速器规格和实测电机数据重新标定。

更直接的限制来自控制尺度和顺序：旧配置所有关节共用 `scale=0.35`，膝和髋的摆幅可能不足；未固定动作顺序时，策略动作可能作用到错误关节。当前代码已经改成逐关节尺度和 canonical order，但这意味着旧 checkpoint 的输出语义不再匹配。

此外还要检查：

- PD `stiffness=80`、`damping=4` 是否与真实执行器响应一致；
- 足底摩擦、质量中心、初始站立姿态和根高；
- 动作变化率、扭矩/速度饱和以及接触相位奖励是否把策略推向原地摆腿。

### 3.2 用数据确认是不是限位

回放或训练时按关节记录以下量：

```text
raw_action, q_target, q_actual, qdot,
distance_to_lower/upper_limit, joint_limit_reward,
applied_torque/current, saturation_flag
```

只有在长时间满足“`raw_action` 接近 ±1、实际角度贴近限位、限位惩罚显著”的情况下，才能把限位认定为主瓶颈。若动作没有触限却步幅小，应先检查动作映射、PD、摩擦、碰撞体和奖励权重。

### 3.3 旧 checkpoint 的处理

以下变化会使旧 checkpoint 不可靠或不兼容：动作尺度改变、动作顺序固定、观测重排、自碰撞设置改变、奖励路由改变。因此 200 轮只学会站立的 checkpoint 不应直接续训；请从当前配置重新训练，并保留新实验的配置快照。

## 4. 问题三：用传感器数据指导训练

### 4.1 推荐数据格式

优先转换为一个时间对齐的 `.npz` 文件。字段如下：

| 字段 | 形状 | 单位/说明 |
| --- | --- | --- |
| `t_s` | `[T]` | 秒，严格递增；缺省时按 `0.02 s` 推断 |
| `joint_pos_rad` | `[T, 8]` | 必需；canonical 右腿后左腿顺序 |
| `joint_vel_rad_s` | `[T, 8]` | 可选 |
| `gyro_rad_s` | `[T, 6, 3]` 或 `[T, 18]` | 可选；六个 IMU 的 body-frame gyro |
| `contact_lr` | `[T, 2]` | 可选；左、右脚接触，`0..1`；力值大于 1 会按 5 N 归一化 |
| `gait_id` | `[T]` | 可选；`0..4`，负值表示不做 gait 过滤 |
| `joint_zero_rad` | `[8]` | 可选；从编码器原始角中减去的零位 |
| `joint_sign` | `[8]` | 可选；每项必须为 `+1` 或 `-1` |

CSV 可使用 `joint_pos_rad[0]` 到 `joint_pos_rad[7]` 等编号列；`.pt/.pth` 和 `.json` 必须包含同名 mapping 字段。加载器会对连续量线性重采样到 50 Hz，对 gait label 使用最近邻。

### 4.2 参考奖励做了什么

启用参考轨迹后，`sensor_reference_imitation` 按可用通道计算指数相似度：

```text
R_ref = weighted_mean(
    exp(-mean(((q - q_ref) / sigma_q)^2)),
    exp(-mean(((qdot - qdot_ref) / sigma_qdot)^2)),
    exp(-mean(((gyro - gyro_ref) / sigma_gyro)^2)),
    exp(-mean(((contact - contact_ref) / sigma_contact)^2))
)
```

缺少某个可选通道时，会从加权平均中移除该通道并重新归一化；没有参考文件时该奖励恒为零，不影响普通训练。默认参考起点固定为 0，避免随机片段在没有 phase 观测时造成非 Markov 奖励。只有在数据已按 gait phase 对齐时才建议打开 `reference_random_start=True`。

注意：参考数据目前是 reward 信号，不会自动把未来参考姿态塞进 actor。actor 仍使用当前仿真可观测量；六个 IMU 如果在场景中可用，会用于 gyro imitation，否则代码会退回根部 gyro。实体端需要先完成 IMU 坐标系和根状态估计，再决定是否重新训练可观测策略。

### 4.3 训练命令

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --num_envs=256 --headless --max_iterations=30000 \
  env.commands.base_velocity.reference_data_path=/absolute/path/walk.npz \
  env.commands.base_velocity.reference_random_start=False \
  env.rewards.sensor_reference_imitation.weight=1.0
```

建议流程：先用 16 个环境运行 2 个 iteration 检查格式和维度，再用 256 个环境做短跑；确认关节顺序、零位、符号和时间戳无误后再增加并行环境。传感器记录若来自不同身高、不同外骨骼或人体关节，应先做长度/坐标/零位 retarget，不能直接逐列复制。

## 5. 五个任务如何验收，以及 checkpoint 重复动作的原因

五个 gait ID 为：

| ID | 模式 | 进入条件 |
| ---: | --- | --- |
| 0 | Stand | 低命令、双脚支撑且根部速度低于阈值持续约 1.5 s |
| 1 | Walk | 常规非零低速命令；第一课程阶段只开放此模式 |
| 2 | Walk-to-Stand | 命令速度范数低于 `0.1 m/s` 后的减速和接触重塑 |
| 3 | Run | 最终课程阶段按 `Fr=v^2/(g*0.8)>0.5` 判定 |
| 4 | Run-to-Walk | 从 Run 降速后稳定约 2.5 s，再回到 Walk |

项目没有独立的 Stand-to-Walk gait ID；Stand 收到非零命令后直接切到 Walk，起步由同一个循环策略学习。

### 5.1 验收步骤

1. 先检查日志中的 `phase`、`gait_mode`、速度命令、实际根部速度和 `fallen`，确认命令确实覆盖目标 gait。
2. 再用 [play_my_exoskeleton.py](scripts/play_my_exoskeleton.py) 回放，每个模式至少记录 10 个 episode；不要只看一段可视化视频。
3. 分别统计：速度误差、严格单支撑接触匹配、双脚支撑时间、腾空脚高度、拖脚速度、跌倒率、关节限位比例和动作饱和比例。
4. 通过一个模式后再提高命令范围；不要在同一轮同时改动作尺度、PD、摩擦和奖励权重。

建议最低验收线：

- Stand：双脚支撑率 ≥ 95%，根部 roll/pitch 绝对值的 95 分位 < 10°，非 timeout 跌倒率 < 5%。
- Walk：命令 `0.2 m/s` 时平均前向速度 `>0.15 m/s`，速度误差 < `0.05--0.10 m/s`，左右脚交替接触而不是长期双支撑。
- W2S：低速命令后在约 1.5 s 内稳定双支撑，不出现持续滑脚。
- Run：只有在确实使用 Run 命令时才验收；应出现相位一致的短支撑/腾空，不能靠站立获得高分。
- R2W：降速后约 2.5 s 内回到 Walk，速度接近 `0.2 m/s` 且不突然跌倒。

### 5.2 为什么打开 checkpoint 后一直重复 W2S/起步动作

常见原因是组合效应：

- 200 轮实验仍在 `WALK_ONLY` 阶段，checkpoint 没有见过完整的 Stand、Run 和 R2W 分布。
- 播放器每次 reset 都重新初始化 gait 状态；如果命令接近零，就会反复进入 W2S。若命令范围固定，也不会自动“继续课程”。
- 当前状态机没有 Stand-to-Walk 标签，视觉上会把 W2S 结束后的起步误认为同一个动作循环。
- LSTM hidden/cell 如果在 episode 边界没有清零，或播放时没有按 checkpoint 的 normalization 统计量处理 49D observation，也会出现周期性重复或失稳。

排查顺序：确认 `gait_mode` 和命令日志，再确认 `running_mean/running_var`、LSTM 状态 reset、动作顺序和 checkpoint 所对应的配置版本。不要用只训练了第一阶段的旧 checkpoint 判断五任务是否学会。

## 6. 500 轮后开放下一个任务的训练策略

当前课程边界由全局控制步计算：

| PPO iteration | 阶段 | 内容 |
| ---: | --- | --- |
| `0--499` | `WALK_ONLY` | 固定 `vx=0.2 m/s`，只训练 Walk |
| `500--999` | `TRANSITIONS` | 加入 W2S/Stand，速度上限 `0.8 m/s` |
| `>=1000` | `MULTI_GAIT` | 开放 Walk、Stand、W2S、Run、R2W，速度上限 `4.0 m/s` |

推荐先运行 1,500 iteration 验证两个边界，再做正式长训：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --num_envs=256 --headless --max_iterations=1500
```

如果从 checkpoint 恢复，RL-Games 不保存 `common_step_counter`，必须补回：

```text
env.curriculum.gait_commands.params.step_offset = 已完成的 PPO iteration * 32
```

旧 200 轮 checkpoint 建议只用于对比，不作为当前配置的初始化权重。逐关节尺度、动作顺序和奖励已经改变，正式训练应从新实验开始。

## 7. 论文生物力学奖励的当前状态

当前配置已经包含以下论文启发的奖励：

- 总质心角动量成本，以及显式左右腿分组的镜像动量协调；
- 关节机械功率 `abs(tau * qdot)` 惩罚；
- 左右腿反相和摆动幅值匹配；
- 支撑腿直膝、相位接触、摆脚高度、拖脚和 Run push-off；
- W2S/R2W 的减速和速度收敛奖励；
- 可选的传感器参考轨迹模仿奖励（关节、gyro、足部接触）；
- 速度、姿态、根高、关节速度/加速度、动作变化率、关节限位和存活项。

论文中的手臂、肩、腰和全身碰撞项不能在只有八个腿部关节的外骨骼上逐字复现，当前使用了明确标注的腿部形态代理。因而只能称为“目标等价或形态适配复现”，不能声称已经复现论文的 Unitree G1 实机结果。完整权重表见 [MY_EXOSKELETON_PAPER_REPRODUCTION.zh-CN.md](source/isaaclab_assets/MY_EXOSKELETON_PAPER_REPRODUCTION.zh-CN.md) 和 [MY_EXOSKELETON_REWARDS.zh-CN.md](scripts/MY_EXOSKELETON_REWARDS.zh-CN.md)。

## 8. 迁移到实体机的必要步骤

### 8.1 实体端必须保持的接口

- 50 Hz 策略线程；底层电机伺服可运行在 200--1000 Hz。
- 输入按仿真相同顺序、单位、坐标系和归一化方式排列。
- 输出是 8 个绝对 joint-position target，不是 torque。实体端必须保存 `q_default` 和新的逐关节 `scale_i`。
- 使用训练 checkpoint 中的 actor observation normalization（`running_mean/running_var`）。
- LSTM hidden/cell 在连续 20 ms 周期间保留；启动、急停、跌倒和故障恢复时显式清零。
- critic 的 62D privileged state 不部署到实体；实体需要按所选任务部署对应 actor 输入：
  flat 基线为 49D，多模态 imitation 为 147D（其中 IMU 54D、RGB 12D、event 32D）。

可先使用 [export_IODescriptors.py](scripts/environments/export_IODescriptors.py) 导出输入输出契约，再实现厂商 SDK bridge 或 `ros2_control` `SystemInterface`。现有 Gazebo `gz_ros2_control` 配置不是实体硬件驱动。

### 8.2 独立安全层

安全层必须位于策略输出和电机控制器之间，并且不依赖 PPO 是否给出合理动作：

- 编码器、IMU、足压有效性和时间戳检查；
- 软限位、速度/加速度 slew-rate 限制；仿真侧的 `3.14 rad/s` 和 `60 N·m` 只用于训练约束，
  不能替代硬件安全上限；
- 电流/力矩、温度、电池、电机故障和通信 watchdog；
- 急停、吊架模式、跌倒检测和故障恢复状态机；
- 超时或传感器异常时切换到经过验证的站立/卸力动作。

URDF 的 `±1.5708 rad`、`3.14 rad/s` 和 `60` effort 不能替代上述硬件限值；必须由真实机构和驱动器规格给出。

### 8.3 分阶段验证，禁止直接穿戴

1. 离线传感器回放：验证单位、零位、符号、时间同步和 observation normalization。
2. 悬空或台架：低增益、限幅动作，确认每个关节方向和急停。
3. 单腿低速测试，再做双腿吊架测试。
4. 先验收 Stand，再以 `0.1--0.2 m/s` 验收 Walk。
5. 最后才测试 W2S/R2W；Run 必须有吊架、限速和急停。

当前仿真没有人体质量、背带柔顺性、人机摩擦和真实足底负载，所以完成上述仿真验收也不等于完成穿戴安全验证。

## 9. 验证记录

本次代码变更已验证：

```bash
uv run python -m compileall -q source/isaaclab_assets/tasks source/isaaclab_assets/envs

OMNI_KIT_ACCEPT_EULA=YES uv run pytest -q \
  source/isaaclab_assets/test/test_my_exoskeleton_env_cfg.py \
  source/isaaclab_assets/test/test_my_exoskeleton_gait.py \
  source/isaaclab_assets/test/test_play_my_exoskeleton.py \
  source/isaaclab_assets/test/test_sensor_reference.py \
  source/isaaclab_assets/test/test_imitation_dataset.py \
  source/isaaclab_assets/test/test_my_exoskeleton_imitation_config.py \
  source/isaaclab_assets/test/test_my_exoskeleton_imitation_runtime.py \
  source/isaaclab_assets/test/test_vision_sensor_cfg.py

uv run ruff check source/isaaclab_assets \
  scripts/train_my_exoskeleton.py scripts/play_my_exoskeleton.py
```

上述命令覆盖配置契约、五步态逻辑、回放入口、参考数据加载、多模态数据集、运行时辅助逻辑和相机契约；本轮 Kit-less 集合结果为 `98 passed`（排除会启动完整 Kit/PhysX 的 `test_valid_configs.py`）。该 Kit 测试需在匹配的 Isaac Sim/GPU 环境中单独运行。以上测试不替代长时间 GPU 训练、实体硬件测试或人体穿戴测试。真实 CAD 限位、执行器响应、IMU 安装姿态、足压标定和人体耦合参数仍需要用户提供并重新验证。

## 10. Vision + IMU + Motion Imitation 环境（当前实现）

### 10.1 注册和文件清单

新的 Gym ID 是 `Isaac-MyExoskeleton-Imitation-v0`，注册在
`source/isaaclab_assets/isaaclab_assets/my_exoskeleton.py`。主要文件如下：

| 文件 | 作用 |
| --- | --- |
| `source/isaaclab_assets/envs/my_exoskeleton_imitation_env_cfg.py` | 继承 `MyExoskeletonFlatEnvCfg`；4096 环境、2.4 m 通道、固定相机、6 个 IMU、观察/奖励组 |
| `source/isaaclab_assets/tasks/my_exoskeleton_imitation.py` | 数据集加载、reset 时任务/窗口采样、IMU 映射、EVIS 运行时桥接 |
| `source/isaaclab_assets/tasks/imitation_dataset.py` | BT/AT trial 发现、50 Hz 同步、CSV/视频/AEDAT4 解码和 validity mask |
| `source/isaaclab_assets/tasks/imitation_reward.py` | IMU、gyro、关节、速度、RGB 特征和 event 特征奖励 |
| `source/isaaclab_assets/tasks/vision_sensor_cfg.py` | 左右 RGB/DVS 固定位姿和 `look_at` 旋转 |
| `source/isaaclab_assets/isaaclab_assets/sensors/event_camera.py` | 没有 EVIS 时的显式无效适配器；绝不使用 RGB 帧差伪造事件 |
| `task_mapping.yaml` / `source/isaaclab_assets/configs/task_mapping.yaml` | 十个任务 ID 和中文名称 |
| `imu_mapping.yaml` / `source/isaaclab_assets/configs/imu_mapping.yaml` | `imu_01`--`imu_06` 到机器人 link 的映射 |
| `sensor_coordinate_mapping.yaml` / `source/isaaclab_assets/configs/sensor_coordinate_mapping.yaml` | 读取层 IMU 轴和髋/膝关节符号校准 |
| `imitation_reward.yaml` / `source/isaaclab_assets/configs/imitation_reward.yaml` | 奖励权重、sigma 和通道契约 |
| `source/isaaclab_assets/configs/my_exoskeleton_imitation_ppo.yaml` | rl_games PPO 配置，包含 4096 actor、32 步 rollout 和 LSTM |

相机固定在 `x=1.2 m,z=1.0 m`，左侧 `y=+1.2 m`、右侧 `y=-1.2 m`，两台 RGB 和两台 event 相机各自使用相同位姿并朝向通道中心。相邻环境间距被强制为至少 `4.0 m`，避免相机看到邻居环境。

### 10.2 任务数据和实际可用性

`/home/lqf/MyCode/Data/records/<subject>/<date>/<task>/TrialXX/` 是当前记录布局。扫描结果为 700 个 metadata trial，其中 696 个包含完整 `imu_01`--`imu_06`，覆盖 BT01--BT05 和 AT01--AT05；4 个缺少六路 IMU 的 trial 被跳过并记录 warning。

每个可用 trial 目前有六路 IMU CSV、左右 RGB MP4、左右 DVXplorer `.aedat4`、事件预览 MP4、压力 CSV 和 metadata JSON。现有记录没有八自由度关节编码器/动作捕捉轨迹，所以 `motion_valid=False`，`imitation_joint` 会被 mask；不能把 IMU 姿态误当成关节角。要启用关节奖励，需额外提供 `[T,8]` 的 `joint_pos_rad`，顺序必须是右腿四关节后左腿四关节。

读取器会在 50 Hz 同步前应用 `sensor_coordinate_mapping.yaml`。八关节 canonical 顺序中，
右髋/右膝/左髋/左膝（索引 `1,2,5,6`）使用符号 `[+1,-1,-1,+1]`，关节速度同步取同一符号；
这与镜像 joint axis 的单关节审计一致，并可由现场 calibration YAML 覆盖。IMU-01/03/05 被标记为左腿
（分别为左踝、左膝、左髋），IMU-02/04/06 被标记为右腿（分别为右踝、右膝、右髋）；腿部姿态、
角速度和加速度都按数据集到 joint 的轴矩阵变换：右腿 `[x_d,y_d,z_d] -> [y_d,x_d,-z_d]`，左腿
`[x_d,y_d,z_d] -> [-y_d,x_d,-z_d]`。IMU-07..11 忽略。该校准只改变数据参考，不改变 URDF 关节轴
或动作控制器；可通过 `env.imitation_calibration_path` 指定现场标定文件。

任务映射：BT01 静态站立标定、BT02 自然行走、BT03 慢速行走、BT04 快速行走、BT05 起步—行走—停止；AT01 转身、AT02 障碍跨越、AT03 斜坡行走、AT04 台阶行走、AT05 手提重物行走。`--task_id BT02` 只选择数据集任务，`--task` 始终是 Gym 环境 ID。未指定任务时，环境支持按 reset 随机抽取任务；同一次 reset 对每个任务只解码一个窗口，避免 4096 次重复视频解码。`BT02/BT03/BT04` 默认允许周期步态窗口循环；`BT01/BT05/AT01-AT05` 默认在窗口末帧保持到 episode reset，可用 `env.imitation_sequence_loop` 和 `env.imitation_non_cyclic_tasks` 覆盖，因此 BT05 不会因回到序列首帧而反复触发“起步—行走—停止”。

当前配置按要求使用 flat terrain 直线通道，因此 AT02--AT04 目前是带任务标签的运动模仿，不会自动生成障碍、斜坡或台阶几何。要验收这些任务的环境交互能力，需要另加对应 terrain/障碍场景和碰撞安全课程；仅凭当前 flat 场景的分数不能宣称已经完成高级物理任务。

### 10.3 传感器和奖励契约

- IMU 观测每路为 `[roll,pitch,yaw,gyro_xyz,accel_xyz]` 共 9 维；原生缓存 `env._latest_imu` 为 `[N,6,9]`，为了 rl_games 的向量拼接，manager 暴露的 `obs["imu"]` 是展平的 `[N,54]`。
- RGB 相机真实输出为 `uint8`，环境在计算特征前转成 `[0,1]`；256×256 原图保留在相机传感器，PPO 只接收两台相机的均值/标准差 12 维 compact feature，避免把 4096×256×256 原图直接送入 MLP。
- event histogram 为 `[window,2 cameras,2 polarities,H,W]`，PPO 对每个时间帧使用 2×4 空间池化得到 32 维特征；时间序列仍由 `window` 目标帧保留。提供的 DVXplorer 源分辨率是 `(height,width)=(240,320)`，先按比例映射到默认 `64×64` 网格。`target_event_stream` 始终保留原生 `(x,y,t,polarity)`，并提供 frame/camera index。
- 没有安装 `dv-processing` 时，AEDAT4 的 `event_valid=False`；没有安装 EVIS `dvs_gen` 时，仿真 event 通道也是无效 mask。两者都不会以 RGB 差分替代。
- 所有模仿相似度都是 `exp(-MSE/sigma²)` 的正分数；缺失通道从权重分母移除。安全、能耗、关节限位等原有 locomotion 项仍保持负权重。完整权重见 `imitation_reward.yaml`。

BT01 静态站立使用 `TASK_SPEED_M_S["BT01"] = 0`。命令项会把这类有限的零速覆盖标记为静态环境，并在 `WALK_ONLY`、`TRANSITIONS` 和 `MULTI_GAIT` 三个课程阶段都保持 `GaitMode.STAND`；不会被固定前向行走命令或 10 Hz 状态更新重新改成 `WALK`。未绑定数据集任务的旧 flat 环境仍使用 `NaN` 覆盖并保持原课程行为。

### 10.4 训练和播放命令

推荐先用小规模 smoke test，再扩大到用户要求的 4096：

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 16 --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data --headless \
  --max_iterations 2
```

通过 smoke test 后的正式命令：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/train_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --num_envs 4096 --task_id BT02 \
  --imitation_dataset /home/lqf/MyCode/Data \
  --max_iterations 30000
```

多任务混合训练可用 `--task_id all`，并保持默认 `env.imitation_random_task_sampling=True`；AT02 等高级任务应在 BT 任务稳定后单独验证。播放命令：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --task_id BT02 --imitation_dataset /home/lqf/MyCode/Data \
  --num_envs 1 --visualizer kit --checkpoint /absolute/path/to/checkpoint.pth
```

播放时可追加 `--show-imu`（只允许 `--num_envs=1 --visualizer kit`），查看六个可选 IMU 探针。RGB 原图可从 `env._latest_camera_rgb` 读取；event 状态和有效标记分别为 `env._latest_event_hist`、`env._latest_event_valid`。viser 适配层应读取这些诊断缓存，不要重新从 RGB 计算事件。

### 10.5 验证顺序和已知运行前提

1. 先运行数据集和配置测试：

   ```bash
   OMNI_KIT_ACCEPT_EULA=YES uv run pytest -q source/isaaclab_assets/test/test_imitation_dataset.py \
     source/isaaclab_assets/test/test_my_exoskeleton_imitation_config.py \
     source/isaaclab_assets/test/test_my_exoskeleton_imitation_runtime.py \
     source/isaaclab_assets/test/test_vision_sensor_cfg.py \
     source/isaaclab_assets/test/test_play_my_exoskeleton.py
   ```

2. 用 `sample_at("BT02", 0.0)` 检查 `target_imu.shape == [W,6,9]`、`target_rgb.shape == [W,2,3,256,256]`、`target_event.shape == [W,2,2,64,64]`，并记录 `motion_valid/rgb_valid/event_valid`。
3. 配置检查应打印 `scene.num_envs=4096`、`scene.env_spacing>=4.0`、四个相机名称和 `cfg.scene.robot.spawn.self_collision=True`。
4. Kit 运行时确认 Stage 中两个 RGB 和两个 DVS prim 的位置、方向，以及 `physxArticulation:enabledSelfCollisions=True`。修改 URDF/STL 后首次启动可加 `env.scene.robot.spawn.force_usd_conversion=True`，避免继续使用旧 USD 缓存。
5. 本工作区的普通 `uv` Python 环境没有 Isaac Sim 的 `carb` 模块；完整 reset/step 必须从 Isaac Sim/Kit Python（或安装 `isaacsim` extra 后的入口）运行。没有 Kit 时只能完成上述无 Kit 测试，不能声称相机、PhysX 自碰撞或 EVIS 已经实机验证。

### 10.6 回放诊断与 Viser

回放入口新增了不会传给 rl_games 的诊断参数。它们由环境按固定间隔输出一行 JSONL，同时在终端记录 IMU、RGB、Event 的有效率和模仿奖励均值：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Imitation-v0 \
  --task_id BT02 --imitation_dataset /home/lqf/MyCode/Data \
  --num_envs 1 --visualizer viser \
  --diagnostics --diagnostics_interval 20 \
  --diagnostics-output /absolute/path/imitation_diagnostics.jsonl \
  --checkpoint /absolute/path/to/checkpoint.pth
```

`camera_rgb_frames` 只在单环境或显式 `store_raw_camera_frames=True` 时保留原图统计；PPO 仍使用 12 维 RGB 特征。`event_camera` 是 EVIS 的真实异步事件直方图，`event_valid_fraction=0` 表示 EVIS 未安装或运行时无效，不表示用 RGB 帧差替代。Viser 负责场景可视化，JSONL 可被外部面板读取；没有 EVIS/dv-processing 时不能把事件诊断解释成真实 DVS 数据。
