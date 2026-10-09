# 外骨骼关节轴与人体膝约束

本文件记录当前 URDF/USD 审计结果和 Isaac Lab 3.0 中已经落地的膝关节安全约束。
它是校准说明，不是实体穿戴安全许可。

## 参考坐标系

| 轴 | 含义 |
| --- | --- |
| `+X` | 外骨骼前进方向 |
| `+Y` | 机器人左侧 |
| `+Z` | 竖直向上 |

源 URDF 的八个 revolute joint 都声明 `axis xyz="0 0 1"`。joint origin 的左右镜像
使膝关节在零位世界坐标中分别约为：

| Joint | 局部轴 | 零位世界轴 | 正角度的几何效果 | 人体屈曲方向 |
| --- | --- | --- | --- | --- |
| `Rthigh_to_Rleg` | `+Z` | `-Y` | 小腿向 `+X` | `q < 0` |
| `Lthigh_to_Lleg` | `+Z` | `+Y` | 小腿向 `-X` | `q > 0` |

髋屈伸的镜像关系同样需要保留：右髋正角度让大腿向前，左髋负角度让大腿向前。
踝关节的解剖正方向取决于脚背/脚底的实际安装零位，当前只保留为可配置符号。

## 人体角与仿真角

定义人体屈曲角 `f` 为正：

```text
f_right_knee = -q_Rthigh_to_Rleg
f_left_knee  = +q_Lthigh_to_Lleg
```

人体训练范围采用 `0 <= f <= 2.2 rad`。仿真中额外保留 `0.05 rad` 的过伸保护余量：

```yaml
Rthigh_to_Rleg: [-2.25, 0.05]
Lthigh_to_Lleg: [-0.05, 2.25]
```

同一组边界同时出现在：

- `source/isaaclab_assets/tasks/robot_contract.py`
- `source/isaaclab_assets/data/my_robot/urdf/my_exoskeleton.urdf`
- `MyExoskeletonRewardsCfg.knee_human_range`
- `MyExoskeletonTerminationsCfg.knee_out_of_human_range`

奖励项 `human_knee_gait_reward` 在支撑期把 `f` 拉向伸直，在摆动期把 `f` 拉向约
`0.8 rad`；`leg_phase_symmetry` 和 `leg_amplitude_match` 负责左右协调。执行器还设置
`joint_velocity_limit=3.14 rad/s` 和 `joint_effort_limit=60 N·m`。

## 数据读取符号

`sensor_coordinate_mapping.yaml` 的八关节符号按 canonical 顺序为：

```text
[base_to_Rturn, Rturn_to_Rthigh, Rthigh_to_Rleg, Rleg_to_Rfoot,
 base_to_Lturn, Lturn_to_Lthigh, Lthigh_to_Lleg, Lleg_to_Lfoot]
[+1, +1, -1, +1, +1, -1, +1, +1]
```

IMU 保持已经验证的左右轴映射：

```text
left : [x_d, y_d, z_d] -> [-y_d,  x_d, -z_d]
right: [x_d, y_d, z_d] -> [ y_d,  x_d, -z_d]
```

其中 IMU-01/03/05 是左踝、左膝、左髋，IMU-02/04/06 是右踝、右膝、右髋，07--11 被忽略。
角速度和加速度使用完整链：

```text
R_exo = R_joint_link @ R_link_sensor @ R_sensor_IMU
```

没有现场安装标定时，前两个矩阵默认为单位阵；不要把单位阵当成已经完成实物安装校准。

## Kit 验收

修改 URDF 后必须确认 USD 重新导入。配置已启用 `force_usd_conversion=True`；旧缓存的
时间戳可能早于 URDF。启动 Kit 后，对每个膝关节发送 `+0.5` 和 `-0.5 rad`，记录：

```text
joint position, joint velocity, target position, applied effort,
distance to lower/upper limit, knee_human_range, knee_out_of_human_range
```

如果右膝正角度让小腿向后、或左膝正角度让小腿向前，说明实体零位/安装方向与当前审计不同。
此时只修改 `KNEE_ARTICULATION_SIGNS`、映射 YAML 和两侧 URDF bounds，并重新做这个试验。

无 Kit 的回归检查：

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0
uv run --no-sync python -m pytest -q source/isaaclab_assets/test \
  --ignore=source/isaaclab_assets/test/test_valid_configs.py
uv run --no-sync ruff check source/isaaclab_assets
```

旧 checkpoint 不应继续用于正式训练：动作符号、URDF 膝限位和奖励项都已经改变。
