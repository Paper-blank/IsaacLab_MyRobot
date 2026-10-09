# `ruleresult` 输出、变时间步长与未来序列

## 1. 先给结论

当前环境的物理步长是 `0.005 s`，`decimation=4`，所以策略控制周期
`control_dt_s=0.02 s`，名义频率为 50 Hz。这个 50 Hz 只描述仿真/控制时钟，
不是主机、网络或电机驱动器能够保证的端到端输出频率。

真实部署中，一次循环还包括策略推理、观测读取、通信、操作系统调度和控制器处理。
如果这些时间合计为 `0.035 s`，下一条命令就可能在约 35 ms 后才发出；开启
`--real-time` 也只能在循环较快时补足等待，不能把超时的一帧变回 20 ms。没有
实时等待时，主机循环甚至可能比仿真时钟更快。

因此：

* `control_dt_s` 是名义控制周期；
* `sample_dt_s`（以及兼容旧版本的 `dt_s`）是本行导数实际使用的相邻采样间隔；
* `output_dt_s` 是相邻 JSONL 输出行的时间间隔；
* `wall_dt_s` 是相邻 JSONL 行的本机单调时钟间隔，不等于电机真正收到命令的时间。

如果需要真实硬件到达时间，应在驱动或通信层采集 `command_sent`、
`actuator_applied` 的硬件时间戳，并通过部署适配层传给记录器。仅凭仿真
`sim.current_time` 不能回答硬件是否晚到。

## 2. 当前模型到底预测什么

现有 PPO actor 的动作契约仍然是：

```text
o_t, h_t -> raw_action_t[8] -> q_target_t[8]
```

其中 `q_target_t` 是经过动作缩放和默认偏置后的八个关节位置目标。环境在收到
这个动作后才产生下一个状态 `s_(t+1)`。当前 actor 没有原生的
`[a_(t+1), ..., a_(t+H)]` 输出头，也没有显式的动力学/world-model 来预测
未来状态。RNN 的 `seq_length` 和 PPO 的 `horizon_length` 是历史记忆及训练采样
参数，不是未来预测长度。

所以，旧 checkpoint 可以可靠输出**当前闭环目标轨迹**的角度、角速度和角加速度，
但不能把一次动作复制多次称为模型预测的未来动作。真正的多步模型预测需要：

1. 给 actor 增加 `H*8` 的多步动作头并重新训练；或
2. 使用动力学仿真/学习到的 world model 做闭环 rollout，并为每个预测状态重新
   计算观测和 recurrent state。

这两种方案都会改变训练或部署接口，不能直接由当前八维 checkpoint 推导出来。

## 3. 时间基准配置

`MyExoskeletonFlatEnvCfg` 新增了以下字段：

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `ruleresult_dt_s` | `0.02` | 名义控制周期及异常时间戳的回退值 |
| `ruleresult_time_mode` | `simulation` | `simulation`、`wall_clock` 或 `configured` |
| `ruleresult_min_dt_s` | `1e-6` | 接受的最小实际间隔 |
| `ruleresult_max_dt_s` | `10.0` | 接受的最大实际间隔 |
| `ruleresult_output_format` | `jsonl` | 磁盘格式：`jsonl` 或 `protocol_55aa` |
| `ruleresult_protocol_config_path` | `None` | `protocol_55aa` 的 8 电机标定文件；协议模式必填 |
| `ruleresult_prediction_horizon` | `0` | 未来序列步数；零表示不输出 |
| `ruleresult_prediction_mode` | `none` | `none`、`kinematic` 或 `supplied` |
| `ruleresult_prediction_dt_s` | `None` | 未来序列的步长；缺省使用控制周期 |

### `simulation`（默认）

优先使用 Isaac Lab 的仿真时间 `sim.current_time`。如果运行时没有这个字段，
则使用 `step * control_dt_s`。这种模式适合比较仿真轨迹，导数不会被主机运行速度
污染；它不能反映硬件命令的墙钟延迟。

### `wall_clock`（实际时间模式）

使用进程的 `time.perf_counter()`（相对记录器初始化附近的时刻）计算：

```text
sample_dt_s = timestamp_s[t] - timestamp_s[t-1]
```

这能暴露推理、通信和调度造成的抖动，适合已经把记录器放在真实控制循环中的部署。
它不是物理仿真时间：只有低于配置的 `ruleresult_min_dt_s` 或超过
`ruleresult_max_dt_s` 的间隔才会被标记为无效并回退。headless 仿真可能产生很小但仍
合法的墙钟间隔；没有实时同步时，不要把 wall-clock 差分直接当成电机速度。

### `configured`

始终按 `step * ruleresult_dt_s` 生成时间戳，适合没有可用时钟的离线回放。

时间戳倒退、太小或超过 `min/max` 范围时，记录器会使用名义周期，置
`time_gap_detected=true`，并清除该段导数历史，避免把跨 episode 或时钟跳变放大成
异常角加速度（某些不推进 `sim.current_time` 的仿真包装器会对零差值采用
`configured_fallback` 兼容处理）。首帧和 reset 后首帧的导数为零，同时
`derivative_history_valid=false`；零值不代表真实静止。

## 4. 非均匀时间间隔下的导数

设目标角度为 `q_k`，实际相邻时间间隔为
`Delta_t_k = t_k - t_(k-1)`。记录器现在使用：

```text
v_k = (q_k - q_(k-1)) / Delta_t_k
a_k = (v_k - v_(k-1)) / (0.5 * (Delta_t_k + Delta_t_(k-1)))
```

因此当实际间隔为 35 ms 时，速度除以 0.035，而不是盲目除以 0.02。加速度对
噪声和时间抖动更敏感；在发送给执行器前仍必须做速度/加速度限幅、软限位、
通信 watchdog 和急停。`ruleresult` 只描述规划量，不替代安全控制器。

## 5. 输出格式

### 5.1 八个关节顺序

所有数组均使用以下固定顺序：

| 索引 | 关节名称 | 机械语义 |
| ---: | --- | --- |
| 0 | `base_to_Rturn` | 右髋转向 |
| 1 | `Rturn_to_Rthigh` | 右髋屈伸 |
| 2 | `Rthigh_to_Rleg` | 右膝 |
| 3 | `Rleg_to_Rfoot` | 右踝 |
| 4 | `base_to_Lturn` | 左髋转向 |
| 5 | `Lturn_to_Lthigh` | 左髋屈伸 |
| 6 | `Lthigh_to_Lleg` | 左膝 |
| 7 | `Lleg_to_Lfoot` | 左踝 |

角度单位为 `rad`，角速度为 `rad/s`，角加速度为 `rad/s^2`。角度是 URDF
带符号关节角，不是转换后的人体屈曲角。

### 5.2 扁平张量

`env.ruleresult_tensor` 保持兼容，形状为 `[N, 24]`：

```text
result[:,  0: 8] = angle_rad
result[:,  8:16] = velocity_rad_s
result[:, 16:24] = angular_acceleration_rad_s2
```

未来序列不会塞进这个 24 维张量；它只通过 JSONL 的 `prediction` 字段或
`env.ruleresult_prediction` 提供。

### 5.3 JSONL 时间字段

每行仍使用 `schema: "ruleresult/v1"`，新增字段是向后兼容的：

| 字段 | 含义 |
| --- | --- |
| `timestamp_s` | 当前选择的时间基准（仿真、墙钟或 configured） |
| `control_dt_s` | 名义控制周期 |
| `sample_dt_s` | 本行导数使用的间隔 |
| `dt_s` | `sample_dt_s` 的旧字段别名 |
| `output_dt_s` | 与上一行 JSONL 的间隔；首行没有此字段 |
| `wall_timestamp_s` | 本机单调时钟的相对时间 |
| `wall_dt_s` | 与上一行记录时刻的墙钟间隔 |
| `dt_source` | `simulation`、`wall_clock`、`configured` 或回退来源 |
| `derivative_valid` | 时间间隔是否通过检查 |
| `derivative_history_valid` | 是否存在可用的上一条同 episode 样本 |
| `time_gap_detected` | 是否发生了时间戳异常/回退 |

`--ruleresult-interval K` 只改变写入文件的行频率，不跳过每个控制步的导数历史。
例如 `K=3` 时，导数仍按每个 `0.02 s` 控制步更新，而相邻文件行的
`output_dt_s` 在仿真模式下约为 `0.06 s`。

示意行（数值仅用于说明字段）：

```json
{
  "schema": "ruleresult/v1",
  "step": 12,
  "episode": 0,
  "env_id": 0,
  "timestamp_s": 0.355,
  "control_dt_s": 0.02,
  "sample_dt_s": 0.035,
  "dt_s": 0.035,
  "output_dt_s": 0.035,
  "wall_timestamp_s": 0.355,
  "wall_dt_s": 0.035,
  "dt_source": "wall_clock",
  "derivative_valid": true,
  "derivative_history_valid": true,
  "time_gap_detected": false,
  "joint_names": ["base_to_Rturn", "Rturn_to_Rthigh", "Rthigh_to_Rleg", "Rleg_to_Rfoot",
                  "base_to_Lturn", "Lturn_to_Lthigh", "Lthigh_to_Lleg", "Lleg_to_Lfoot"],
  "angle_rad": [0.0, 0.1, 0.2, 0.0, 0.0, -0.1, 0.2, 0.0],
  "velocity_rad_s": [0.0, 2.86, 5.71, 0.0, 0.0, -2.86, 5.71, 0.0],
  "angular_acceleration_rad_s2": [0.0, 18.0, 32.0, 0.0, 0.0, -18.0, 32.0, 0.0],
  "raw_action": [0.0, 0.15, 0.21, 0.0, 0.0, -0.15, 0.21, 0.0],
  "measured": {
    "angle_rad": [0.0, 0.09, 0.18, 0.0, 0.0, -0.09, 0.18, 0.0],
    "velocity_rad_s": [0.0, 0.45, 0.92, 0.0, 0.0, -0.46, 0.91, 0.0],
    "angular_acceleration_rad_s2": [0.0, 2.1, 4.5, 0.0, 0.0, -2.0, 4.1, 0.0]
  },
  "prediction_status": "disabled"
}
```

`angle_rad`、`velocity_rad_s` 和 `angular_acceleration_rad_s2` 是策略位置目标的
规划量；`measured` 是执行器/仿真器状态，不能混为一谈。

### 5.4 实体通讯采用 8 电机 `55AA` 二进制帧

`ruleresult/v1` JSONL 是便于检查、插值和复现实验的**规划数据格式**，不是电机直接
接收的线路格式。实体部署层会在完成关节校准和安全检查后，把同一个 8 关节样本
编码为固定 53 字节的 `55AA` 控制帧。JSONL 格式和 actor 的 8 维输出保持不变，旧
checkpoint 不需要因为通讯协议而重新训练。

二进制帧的固定结构为：

```text
55 AA | 32 | 8 * (目标位置 uint16 + 带方向目标速度 uint16 + 加速度幅值/上限 uint16)
      | XOR | 5A

总帧长：2 + 1 + 48 + 1 + 1 = 53 B
```

其中 `0x32` 表示其后的 50 字节（48 字节数据、1 字节校验、1 字节帧尾），不是十进制
32。三个 `uint16` 均采用大端序（big-endian，高字节在前，例如 `0x1234 → 12 34`）。
这与接收固件的 `(high_byte << 8) | low_byte` 一致。XOR 覆盖帧头、长度和全部
48 字节数据，即完整帧
偏移 0--50；不包含校验字节和帧尾。全零数据的校验值为 `0xCD`。

协议槽位顺序与上面的 JSONL 数组顺序不同，必须按关节名称重排：

| 协议槽位 | 关节名称 | 机械语义 |
| ---: | --- | --- |
| 1 | `Lleg_to_Lfoot` | 左踝 |
| 2 | `Rleg_to_Rfoot` | 右踝 |
| 3 | `Lthigh_to_Lleg` | 左膝 |
| 4 | `Rthigh_to_Rleg` | 右膝 |
| 5 | `Lturn_to_Lthigh` | 左髋屈伸 |
| 6 | `Rturn_to_Rthigh` | 右髋屈伸 |
| 7 | `base_to_Lturn` | 左髋转向 |
| 8 | `base_to_Rturn` | 右髋转向 |

**通俗解释：** 线路中的三个数字虽然都存为 `uint16`，但角度和速度会减掉中点偏移，
因此仍表示正负方向。加速度没有中点偏移，只表示大小或上限。原“最大速度”名称
不适合用户提供的接收公式：速度必须保留轨迹方向，不能先取绝对值。

**技术细节：** 关节量只做一次标定：

```text
q_motor = zero_offset_rad + direction * gear_ratio * q_joint
v_motor =                   direction * gear_ratio * v_joint
a_magnitude = abs(gear_ratio * a_joint)

q_u16 = round_half_up((q_motor + 12.57) / 25.14 * 65535)
v_u16 = round_half_up((v_motor + 20.0) / 40.0 * 65535)
a_u16 = round_half_up(a_command_rad_s2 * 1000)

q_motor = q_u16 / 65535 * 25.14 - 12.57      # -12.57..+12.57 rad
v_motor = v_u16 / 65535 * 40.0 - 20.0       # -20..+20 rad/s
a_command_rad_s2 = a_u16 / 1000.0          # 0..65.535 rad/s²
```

`round_half_up` 为半整数向上取整。**加速度解析是基于给定量程补齐的约定，仍需与
固件核对**，用户提供的 C 片段尚未包含这一项。`position_min_rad/max_rad` 现在仅
作为电机侧实测软限位，不再改变固定位置量程；协议 `±12.57 rad` 不是人体机械
安全限位。速度/加速度满量程固定为 `20.0` 和 `65.535`，另有审核过的更保守安全上限。

`limit_source=trajectory` 默认发送轨迹加速度幅值；`configured` 发送审核过的固定
加速度上限。**两种模式的速度始终来自带符号轨迹**；`command_max_velocity_rad_s`
仅检查真实速度幅值，不能用来替代目标速度。若填写了速度/加速度命令上限，则原始
轨迹导数也必须通过检查；任何越界或非有限数都拒绝编码，不溢出或静默截断。

注意角度/速度无法精确表示零点：零目标编码为 `0x8000`，反解约为
`+0.000191806 rad` 和 `+0.000305180 rad/s`。**48B 全零数据不是停止命令**，反解为
每电机 `-12.57 rad`、`-20 rad/s`、`0 rad/s²`，只能用于断能协议测试。

编码器在量化前检查原始轨迹，还会把量化整数按固件公式反解，再次检查软位置范围、
速度/加速度上限；反解越界时拒绝输出，不静默裁剪（fail-closed）。轨迹与审核上限间
须预留量化和浮点余量；例如 `q=0` 在软上限 `0` 下，或加速度 `2.0006` 在同值上限
下，量化后可能超过边界并失败。速度上限设 `0` 不是停机指令，零速度量化后并不为零，
会因超出零上限而拒绝。Python 64 位浮点与固件32位 `float` 舍入也可能不同，接收端
须对实际解析值再次安全检查，不可只依赖导出时检查。

完整字节偏移、量化公式、校验示例和接收规则见项目根目录
`communication_protocol.md`。配置格式见
`scripts/deployment/protocol_55aa_config.example.json`，schema 为
`exoskeleton-55aa-config/v2`、`byte_order=big`；该文件只有未标定的示例值，
且 `example_only: true` 会被加载器主动拒绝，不能直接产生二进制输出。必须复制模板、
替换 8 个电机的全部实测参数并完成复核，然后在副本中将 `example_only` 改为 `false`；
不得只改标志而继续使用示例数字。
`slot`、`direction` 必须是严格 JSON 整数（分别为 `1..8`、`±1`），布尔值、浮点值
和字符串均被拒绝；`1.0` 不能代替 `1`。

选择 `protocol_55aa` 时，同一次运行每个输出周期向 `.bin` 文件追加一帧 53 字节数据；
新打开输出时会拒绝已有非空文件，保持旧内容不变，必须用新路径（已有空文件允许）。文件中
没有 JSON 文本、时间戳或行分隔符。记录器内存中的 `env.ruleresult` 仍保留完整的
`ruleresult/v1` 字典，便于运行时诊断。若需要同时长期保存可读 JSONL 和线路帧，
应由部署程序分别记录，不能把两种格式写进同一个文件。

旧 `v1` 标定（小端、可变位置量程、非负速度上限）会被加载器明确拒绝；不能仅改
schema 就视为完成迁移。旧 `.bin` 与新帧同长且没有版本字段，XOR 无法区分二者，
必须隔离旧文件、从可溯源 JSONL 重新导出到新路径，禁止混合追加或误播放旧帧。

若线路使用 CAN FD，53 B 不能直接作为 DLC 长度，应使用 64 B DLC并补11 B约定填充。
接收传输适配层先剥离填充，再将53 B交给协议解码器；填充不改变长度字节或 XOR，
也不写入本记录器的 `.bin` 文件。

## 6. 未来序列输出

### 6.1 `none`

默认模式，不输出 `prediction`。这是当前旧 checkpoint 的诚实表示：actor 只给出
当前一步动作。

### 6.2 `kinematic`

这是兼容旧模型的短期运动学外推，不是 actor 的未来动作预测。对当前规划目标
`q_target` 及其规划导数 `v_target`、`a_target`，在未来时间 `tau` 使用：

```text
q_future = q + v * tau + 0.5 * a * tau^2
v_future = v + a * tau
a_future = a
```

刚 reset 后没有有效的 `v_target/a_target` 时，外推会暂时保持当前目标角度；
这只是缺少历史信息的保守行为，不表示 actor 预测了未来状态。

JSON 中 `prediction.source` 为
`"constant_acceleration_extrapolation"`，并包含：

```text
prediction.horizon_steps
prediction.timestamps_s          # 相对当前样本的正时间戳
prediction.dt_s
prediction.angle_rad[H][8]
prediction.velocity_rad_s[H][8]
prediction.angular_acceleration_rad_s2[H][8]
prediction.valid_mask[H]
```

### 6.3 `supplied`

用于外部 planner、world model 或真正的多步 actor。通过
`env.set_ruleresult_prediction_callback(callback)` 注入回调：

```python
def callback(env, current_targets, horizon, dt_s):
    # 返回 [N, H, 8] 的未来角度目标；也可返回 [H, 8]（会广播到 batch）。
    return future_angle_targets

env.set_ruleresult_prediction_callback(callback)
```

记录器会按每个未来时间戳计算速度和角加速度，并把来源写入
`prediction.source`。没有回调时 `prediction_status` 为 `"unavailable"`，不会复制
当前动作伪造未来序列。

也可以离线使用公共函数：

```python
from isaaclab_assets.tasks.ruleresult import predict_ruleresult_sequence

result = predict_ruleresult_sequence(
    angle_targets_rad,       # [H, 8] 或 [N, H, 8]
    timestamps_s=future_time,
)
```

该函数接受非均匀未来时间戳；`extrapolate_ruleresult_sequence` 提供上述恒加速度
外推。两者都只做运动学计算，不调用 PPO actor 或模拟器。

## 7. 启用方式

基础 JSONL 输出：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Flat-v0 \
  --checkpoint latest --num_envs=1 --headless \
  --ruleresult --ruleresult-output logs/ruleresult.jsonl
```

输出 8 电机 `55AA` 二进制帧：

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --extra rl-games \
  python scripts/play_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Flat-v0 \
  --checkpoint latest --num_envs=1 --headless \
  --ruleresult \
  --ruleresult-format protocol_55aa \
  --ruleresult-protocol-config /absolute/path/to/reviewed_protocol_55aa_config.json \
  --ruleresult-output logs/ruleresult.bin
```

只提供 `--ruleresult-protocol-config` 时也会自动选择 `protocol_55aa`。若没有显式指定
`--ruleresult-output`，协议模式的默认文件名是 `ruleresult.bin`，JSONL 模式仍是
`ruleresult.jsonl`。`scripts/deployment/protocol_55aa_config.example.json` 只能用于理解
字段；它会被加载器故意拒绝。必须复制为现场配置，用实测值替换全部标定项并完成
复核，再将副本中的 `example_only` 改为 `false`，才可生成协议文件并进入硬件评审流程。
现场配置必须为 `v2` 大端版本；速度/加速度固定满量程不是需要测量替换的示例参数。
要用实测数据替换的是软限位、零位、方向、减速比和命令安全上限。

命令行给出的相对输出路径（例如 `logs/ruleresult.jsonl`）会在启动时解析为
**启动目录下的绝对路径**，因此不会因为加载 checkpoint 后 `env.cfg.log_dir` 被设置为
训练运行目录而悄悄移动到另一个 `logs/` 子目录。程序首次打开文件时也会打印最终路径；
如果没有指定 `--ruleresult-output`，则仍沿用默认的 checkpoint 运行目录日志路径。

按实际墙钟间隔计算规划量，并输出四步恒加速度外推：

```bash
python scripts/play_my_exoskeleton.py \
  --task Isaac-MyExoskeleton-Flat-v0 --checkpoint latest --num_envs=1 --headless \
  --ruleresult --ruleresult-use-actual-dt \
  --real-time \
  --ruleresult-prediction-mode kinematic \
  --ruleresult-prediction-horizon 4 \
  --ruleresult-prediction-dt 0.02 \
  --ruleresult-output logs/ruleresult_wall.jsonl
```

等价的 Hydra 覆盖写法是：

```text
env.ruleresult_time_mode=wall_clock
env.ruleresult_prediction_mode=kinematic
env.ruleresult_prediction_horizon=4
env.ruleresult_prediction_dt_s=0.02
```

也可以直接指定时间模式：

```bash
--ruleresult-time-mode simulation
--ruleresult-time-mode wall_clock
--ruleresult-time-mode configured
```

`wall_timestamp_s` 和 `wall_dt_s` 在记录器完成一次 `env.step` 后采样，主要用于
发现主机循环和文件写出的延迟；它们不是总线发送或执行器到达时间戳。
JSONL 输出文件按追加模式打开；重复使用同一路径时请先归档旧文件或指定新路径。真实
硬件使用时还应由驱动层提供发送/到达时间戳，并在安全控制器中执行限幅和急停。
二进制模式在同次运行内追加，但禁止打开已有非空文件；可用“文件大小必须为 53 的整数倍”做最基本的完整性
检查，但这不能代替逐帧帧头、长度、XOR 和帧尾校验。跨越本次小端→大端升级时
必须使用新路径，不能对已有的旧二进制文件继续追加。
