# 双足机器人足底压力传感器说明

本文记录本次足底压力传感器改动，面向需要启动、验证或继续修改该功能的开发者。

## 1. 当前配置

| 项目 | 当前值 |
| --- | --- |
| 仿真器 | Gazebo Harmonic 8.11.0 |
| ROS 2 | Jazzy |
| 左脚 link | `Lfoot` |
| 右脚 link | `Rfoot` |
| 左脚碰撞体 | `Lfoot_collision` |
| 右脚碰撞体 | `Rfoot_collision` |
| 每只脚 ContactSensor 数量 | 1 个 |
| 虚拟压力点数量 | 60 个，10 列 × 6 行 |
| 足底公开输出频率 | 20 Hz |
| IMU 公开输出频率 | 200 Hz |

重要：当前并不是在 Gazebo 中创建 60 个独立的物理传感器。每只脚使用一个
Gazebo `ContactSensor` 获取实际接触点和接触力，再由 ROS 节点把接触点映射到
固定的 10×6 网格中。因此，60 个数组元素表示 60 个虚拟网格单元。

## 2. 文件结构和职责

主机 ROS 工作空间：

```text
/home/lqf/MyCode/my_exoskeleton_ros2_ws/
├── PRESSURE_SENSOR_GUIDE_ZH.md       # 本说明
└── src/my_exoskeleton_description/
    ├── launch/gazebo.launch.py       # Gazebo、桥接和 ROS 节点启动
    ├── worlds/empty.sdf               # Harmonic world 和 Gazebo 系统插件
    ├── scripts/contact_grid_mapper.py # 接触点 -> 60 点压力网格
    ├── scripts/sensor_rate_limiter.py # 对外话题限频
    ├── CMakeLists.txt                 # 安装脚本，并生成 Gazebo model.sdf
    └── package.xml                    # ROS 2 运行依赖
```

URDF 的 canonical 源文件在 Isaac Lab 目录，不在 ROS 包中直接维护：

```text
/home/lqf/MyCode/IsaacLab-v3.0.0/source/isaaclab_assets/data/my_robot/urdf/my_exoskeleton.urdf
```

容器内对应路径通常为：

```text
/workspace/isaaclab/source/isaaclab_assets/data/my_robot/urdf/my_exoskeleton.urdf
/workspace/ros2_ws
```

修改 URDF 后必须重新构建 ROS 工作空间。构建时会把 URDF 转换为 Gazebo 使用的
`model.sdf`，并安装到 ROS 工作空间的 `install` 目录；不要直接修改生成的
`install/.../model.sdf`。

## 3. 传感器配置

源 URDF 中的左右脚配置位于 `<gazebo reference="Lfoot">` 和
`<gazebo reference="Rfoot">`：

```xml
<gazebo reference="Lfoot">
  <selfCollision>true</selfCollision>
  <max_contacts>10</max_contacts>
  <min_depth>0.001</min_depth>
  <sensor name="left_foot_contact_sensor" type="contact">
    <always_on>true</always_on>
    <update_rate>20</update_rate>
    <visualize>false</visualize>
    <contact>
      <collision>Lfoot_collision</collision>
      <topic>/left_foot_contact</topic>
    </contact>
  </sensor>
</gazebo>

<gazebo reference="Rfoot">
  <selfCollision>true</selfCollision>
  <max_contacts>10</max_contacts>
  <min_depth>0.001</min_depth>
  <sensor name="right_foot_contact_sensor" type="contact">
    <always_on>true</always_on>
    <update_rate>20</update_rate>
    <visualize>false</visualize>
    <contact>
      <collision>Rfoot_collision</collision>
      <topic>/right_foot_contact</topic>
    </contact>
  </sensor>
</gazebo>
```

`worlds/empty.sdf` 启用了 `gz-sim-contact-system`。此外，`CMakeLists.txt` 会把
机器人作为 world 启动时的静态 model include 加载，这样 Contact system 能在
模型载入阶段注册传感器。

## 4. ROS 话题

### 4.1 足底压力话题

| 话题 | 类型 | 频率 | 内容 |
| --- | --- | --- | --- |
| `/left_foot_contact` | `ros_gz_interfaces/msg/Contacts` | 20 Hz | 左脚公开接触数据 |
| `/right_foot_contact` | `ros_gz_interfaces/msg/Contacts` | 20 Hz | 右脚公开接触数据 |
| `/left_foot_pressure_grid` | `std_msgs/msg/Float64MultiArray` | 20 Hz | 左脚 60 点法向力网格 |
| `/right_foot_pressure_grid` | `std_msgs/msg/Float64MultiArray` | 20 Hz | 右脚 60 点法向力网格 |

`/left_foot_pressure_grid` 和 `/right_foot_pressure_grid` 的 `data` 数组长度固定
为 60，索引方式为：

```text
index = y_index * 10 + x_index
```

`y_index` 范围是 0 到 5，`x_index` 范围是 0 到 9。每个元素是该网格单元累加
后的法向力，单位通常为牛顿（N）。无接触时发布 60 个零值。

### 4.2 原始话题和限频

Gazebo Contact/IMU 输出可能接近物理仿真步频，不能只依赖 URDF 的
`update_rate` 来保证 ROS 话题端的实际频率。因此启动文件把桥接后的原始数据
重映射到 `_raw` 话题，再由 `sensor_rate_limiter.py` 定时发布公开话题：

```text
/my_exoskeleton/imu/lfoot_raw  -> /my_exoskeleton/imu/lfoot       200 Hz
/left_foot_contact_raw         -> /left_foot_contact              20 Hz
```

同样的规则适用于 6 路 IMU 和左右脚 Contact。`_raw` 话题用于调试，公开话题
用于控制器、记录节点和压力网格节点。

## 5. 快速启动

下面命令在 Isaac Lab ROS 2 容器内执行：

```bash
source /opt/ros/jazzy/setup.bash
cd /workspace/ros2_ws

colcon build --symlink-install \
  --cmake-args \
  -DISAACLAB_ASSET_DIR=/workspace/isaaclab/source/isaaclab_assets/data/my_robot

source install/setup.bash
ros2 launch my_exoskeleton_description gazebo.launch.py headless:=false
```

如果不需要图形界面：

```bash
ros2 launch my_exoskeleton_description gazebo.launch.py headless:=true
```

从主机直接执行时：

```bash
docker exec -it isaac-lab-ros2-lqf bash
```

进入容器后再执行上面的构建和启动命令。GUI 模式还需要先完成 X11、
`DISPLAY` 和 Xauthority 配置，具体参考工作空间中的 `README_zh.md`。

## 6. 验证传感器

启动后，先确认话题存在：

```bash
ros2 topic list | grep -E 'foot_contact|pressure_grid|imu/'
```

验证左右脚公开 Contact 频率：

```bash
ros2 topic hz /left_foot_contact
ros2 topic hz /right_foot_contact
```

预期约为 `20 Hz`。验证任意一路公开 IMU：

```bash
ros2 topic hz /my_exoskeleton/imu/lfoot
```

预期约为 `200 Hz`。检查压力网格长度和数据：

```bash
ros2 topic echo /left_foot_pressure_grid --once
ros2 topic echo /right_foot_pressure_grid --once
```

输出应包含 6 行、10 列的布局信息，并且 `data` 中有 60 个浮点数。

确认话题发布者：

```bash
ros2 topic info -v /left_foot_contact
ros2 topic info -v /my_exoskeleton/imu/lfoot
```

公开话题的发布者应为 `sensor_rate_limiter`，对应 `_raw` 话题的发布者应为
`ros_gz_bridge`。

## 7. 常见修改位置

### 修改传感器频率

1. 在源 URDF 中修改 ContactSensor 的 `<update_rate>`，或修改 6 路 IMU 的
   `<update_rate>`。
2. 在 `sensor_rate_limiter.py` 中同步修改定时器：

   ```python
   self.create_timer(1.0 / 200.0, self._publish_imu)
   self.create_timer(1.0 / 20.0, self._publish_contacts)
   ```

3. 重新 `colcon build` 并重新启动 Gazebo。

仅修改 URDF 不一定能保证 ROS 端最终输出频率；公开话题由限频节点的定时器
决定。

### 修改 10×6 网格尺寸

在 `contact_grid_mapper.py` 中同步修改：

```python
self.rows = 6
self.cols = 10
```

并检查 `_grid_message()` 的 `stride`、网格边界和下游节点是否仍然按新的数组
长度读取。网格长度始终是 `rows * cols`。

### 修改脚底空间范围

`contact_grid_mapper.py` 的 `self.bounds` 定义每只脚局部坐标的映射范围：

```python
self.bounds = {
    "left": (x_min, x_max, y_min, y_max),
    "right": (x_min, x_max, y_min, y_max),
}
```

接触点超出范围时会被忽略。更换 STL、碰撞网格或坐标系后，需要重新测量并
更新这里的范围。

### 修改话题名称

需要同时修改以下三处：

1. URDF 中 ContactSensor 的 `<topic>`。
2. `gazebo.launch.py` 中 `contact_bridge` 的 Gazebo 话题和 remapping。
3. `sensor_rate_limiter.py`、`contact_grid_mapper.py` 中的订阅和发布话题。

只修改其中一处会导致话题存在但没有数据。

## 8. 故障排查

### 话题不存在

检查 Gazebo 是否加载了 Contact system 和机器人 model：

```bash
ros2 topic list | grep foot_contact
```

同时查看启动日志中是否出现左右脚 ContactSensor 的注册信息。如果没有，先
检查 `gz-sim-contact-system`、`GZ_SIM_RESOURCE_PATH` 和生成的 model 文件。

### Contact 有数据，但压力网格全为 0

先区分是没有接触，还是坐标变换失败：

```bash
ros2 topic echo /left_foot_contact --once
ros2 run tf2_ros tf2_echo world Lfoot
ros2 run tf2_ros tf2_echo world Rfoot
```

如果 Contact 消息中没有 `contacts`，机器人脚底没有接触地面或碰撞参数需要
调整。如果 Contact 有位置和力，但 mapper 日志提示无法从 `world` 转换到脚部
坐标系，需要检查 `gazebo.launch.py` 中的 `left_foot_frame`、
`right_foot_frame` 是否与实际 TF 帧名一致。

启动文件的 `base_z` 参数是压力映射使用的固定 `world -> base_link` 初始参考，
默认值为 `0.0`：

```bash
ros2 launch my_exoskeleton_description gazebo.launch.py \
  headless:=true base_z:=0.0
```

如果场景使用固定的其他出生高度，可以传入对应值。机器人存在浮动基座并且会
移动时，应改为提供动态 `world -> base_link` TF，而不是长期使用静态 TF。

### 频率不符合预期

分别检查公开和原始话题：

```bash
ros2 topic hz /left_foot_contact
ros2 topic hz /left_foot_contact_raw
ros2 topic hz /my_exoskeleton/imu/lfoot
ros2 topic hz /my_exoskeleton/imu/lfoot_raw
```

公开话题目标是足底 20 Hz、IMU 200 Hz；原始话题可能接近 Gazebo 物理步频，
不作为控制接口的频率依据。

## 9. 修改后的标准流程

```text
修改 Isaac Lab 源 URDF 或 ROS 脚本
            |
            v
容器内 colcon build --symlink-install
            |
            v
source install/setup.bash
            |
            v
重新启动 gazebo.launch.py
            |
            v
ros2 topic hz + ros2 topic echo 验证频率和数据
```

不要删除 `build`、`install`、Docker 容器或卷来解决普通的频率/话题问题；优先
重新构建并检查日志。任何删除操作都应先确认目标和备份需求。

