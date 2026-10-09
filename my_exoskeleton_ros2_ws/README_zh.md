# Isaac Lab ROS 2 可视化启动说明

本文说明如何使用本地镜像 `isaac-lab-ros2-lqf:latest` 启动带有图形界面的
Isaac Lab、Isaac Sim 和 Gazebo Harmonic。

## 1. 当前环境

| 项目 | 路径或名称 |
| --- | --- |
| Isaac Lab 主机目录 | `/home/lqf/MyCode/IsaacLab-v3.0.0` |
| ROS 2 工作空间主机目录 | `/home/lqf/MyCode/my_exoskeleton_ros2_ws` |
| Docker 镜像 | `isaac-lab-ros2-lqf:latest` |
| Docker 容器 | `isaac-lab-ros2-lqf` |
| 容器内 Isaac Lab | `/workspace/isaaclab` |
| 容器内 ROS 2 工作空间 | `/workspace/ros2_ws` |
| ROS 2 版本 | Jazzy |
| Gazebo 版本 | Harmonic（`gz sim 8.11.0`） |

工作空间和 Isaac Lab 源码是两个独立的挂载：

```text
/home/lqf/MyCode/my_exoskeleton_ros2_ws
        -> /workspace/ros2_ws

/home/lqf/MyCode/IsaacLab-v3.0.0/source
        -> /workspace/isaaclab/source
```

URDF 和 meshes 的 canonical 源文件位于：

```text
/workspace/isaaclab/source/isaaclab_assets/data/my_robot/
├── urdf/my_exoskeleton.urdf
└── meshes/
```

修改 URDF 或 meshes 后，需要在容器内重新执行 `colcon build`。

## 2. 启动前检查

必须在宿主机的图形桌面终端执行下面的命令。宿主机应当有非空的
`DISPLAY` 和可读的 Xauthority 文件：

```bash
echo "$DISPLAY"
echo "$XAUTHORITY"
test -d /tmp/.X11-unix
test -r "$XAUTHORITY"
```

当前机器的典型输出为：

```text
DISPLAY=:1
XAUTHORITY=/run/user/1000/gdm/Xauthority
```

如果 `DISPLAY` 为空，请先登录 Ubuntu 图形桌面；通过没有图形会话的 SSH
终端不能直接显示窗口。

## 3. 用现有镜像创建带 X11 的容器

下面的命令只使用现有的 `isaac-lab-ros2-lqf:latest` 镜像，不执行 Docker
build，也不会删除命名卷。它会重新创建容器，以加入 X11 socket 和认证文件
挂载。

```bash
cd /home/lqf/MyCode/IsaacLab-v3.0.0/docker

export DOCKER_NAME_SUFFIX=-lqf
export DISPLAY="${DISPLAY:?请在图形桌面终端中执行此命令}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
export __ISAACLAB_TMP_XAUTH="$XAUTHORITY"
export __ISAACLAB_TMP_DIR="$(dirname "$XAUTHORITY")"

test -r "$XAUTHORITY"
test -d /tmp/.X11-unix

docker compose \
  --file docker-compose.yaml \
  --file /home/lqf/MyCode/my_exoskeleton_ros2_ws/docker-compose.isaaclab.yaml \
  --file x11.yaml \
  --profile ros2 \
  --env-file .env.base \
  --env-file .env.ros2 \
  up --detach --no-build --force-recreate isaac-lab-ros2
```

注意：这里没有使用 `down --volumes`，因此 Isaac Sim 缓存、日志和
`/workspace/isaaclab/data_storage` 等命名卷会保留。

验证 X11 挂载是否生效：

```bash
docker exec isaac-lab-ros2-lqf bash -lc \
  'echo "DISPLAY=$DISPLAY"; \
   echo "XAUTHORITY=$XAUTHORITY"; \
   test -S "/tmp/.X11-unix/X${DISPLAY#:}" && echo "X11 socket: OK"'
```

预期结果应包含 `DISPLAY=:1`（或当前显示编号）和 `X11 socket: OK`。

## 4. 进入容器

```bash
docker exec -it isaac-lab-ros2-lqf bash
```

进入容器后可以检查：

```bash
echo "$DISPLAY"
source /opt/ros/jazzy/setup.bash
export LD_LIBRARY_PATH="/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:${LD_LIBRARY_PATH:-}"
gz sim --versions
```

## 5. 在容器内部打开 GUI

完成第 3 节的 X11 配置后，先进入正在运行的容器：

```bash
docker exec -it isaac-lab-ros2-lqf bash
```

进入容器后确认显示变量和 X11 socket：

```bash
echo "$DISPLAY"
test -S "/tmp/.X11-unix/X${DISPLAY#:}" && echo "X11 socket: OK"
```

在当前 shell 中启动 Isaac Sim 或 Isaac Lab 前，先加载 ROS 2 环境以及
Isaac Sim 内置的 ROS 2 Bridge 库：

```bash
source /opt/ros/jazzy/setup.bash
export LD_LIBRARY_PATH="/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:${LD_LIBRARY_PATH:-}"
```

其中第二行用于让 Isaac Sim 找到 `librcutils.so` 等 ROS 2 动态库，避免
出现 `ROS2 Bridge startup failed`。

### 5.1 直接启动 Isaac Sim

如果只需要打开 Isaac Sim 的空白图形界面，在容器内执行：

```bash
cd /workspace/isaaclab
./isaaclab.sh -s
```

该命令直接调用镜像内的 `isaac-sim.sh`，窗口关闭后命令结束。

### 5.2 启动 Isaac Lab 可视化示例

在容器内执行一个自带的 Isaac Lab 示例：

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

该示例会打开 Isaac Sim 窗口。关闭窗口或按 `Ctrl+C` 可以结束示例。

也可以运行自己的 Isaac Lab Python 脚本：

```bash
cd /workspace/isaaclab
./isaaclab.sh -p <你的脚本路径.py>
```

`isaaclab.sh` 会调用镜像中与 Isaac Lab 配套的 Isaac Sim Python 环境。

### 5.3 不进入交互 shell 直接启动

也可以从宿主机直接在容器内启动 GUI 程序。GUI 所需的 `DISPLAY`、X11
socket 和 Xauthority 必须已经按照第 3 节配置到容器：

```bash
docker exec -it isaac-lab-ros2-lqf bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   export LD_LIBRARY_PATH="/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:${LD_LIBRARY_PATH:-}" && \
   cd /workspace/isaaclab && ./isaaclab.sh -s'
```

直接运行自己的脚本：

```bash
docker exec -it isaac-lab-ros2-lqf bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   export LD_LIBRARY_PATH="/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:${LD_LIBRARY_PATH:-}" && \
   cd /workspace/isaaclab && ./isaaclab.sh -p <你的脚本路径.py>'
```

关闭 Isaac Sim 窗口或在终端按 `Ctrl+C` 可以结束程序。

## 6. 启动 Gazebo Harmonic 图形界面

先构建并加载 ROS 2 工作空间：

```bash
source /opt/ros/jazzy/setup.bash
cd /workspace/ros2_ws
colcon build --symlink-install \
  --cmake-args \
  -DISAACLAB_ASSET_DIR=/workspace/isaaclab/source/isaaclab_assets/data/my_robot
source install/setup.bash
```

启动 Gazebo GUI：

```bash
ros2 launch my_exoskeleton_description gazebo.launch.py headless:=false
```

启动文件会完成以下工作：

1. 启动 Gazebo Harmonic GUI。
2. 在 `empty` world 中生成 `my_exoskeleton`。
3. 启动 `robot_state_publisher`。
4. 加载 `gz_ros2_control`。
5. 桥接 Gazebo 的 `/clock` 到 ROS 2。
6. 激活 `joint_state_broadcaster` 和 `exoskeleton_position_controller`。

检查控制器：

```bash
ros2 control list_controllers -c /controller_manager
```

预期两个控制器的状态都是 `active`。

不带 `headless:=false` 时，启动文件默认使用 server-only 模式，不会打开
Gazebo 窗口：

```bash
ros2 launch my_exoskeleton_description gazebo.launch.py
```

## 7. 日常停止和再次启动

停止容器但保留容器和所有命名卷：

```bash
docker stop isaac-lab-ros2-lqf
```

再次启动已经配置好 X11 的容器：

```bash
docker start isaac-lab-ros2-lqf
docker exec -it isaac-lab-ros2-lqf bash
```

如果宿主机的 Xauthority 路径发生变化，请重新执行第 3 节的 Compose 命令
更新容器挂载。

## 8. 关于 `container.py start`

对于当前这个已经存在的 ROS 2 镜像，不建议每次使用下面的命令启动：

```bash
python3 docker/container.py start ros2 ...
```

当前 `container.py start` 对派生 profile 会先无条件构建
`isaac-lab-base`。`.env.base` 中配置的是 Isaac Sim `6.0.0`，如果本地
没有该上游层，Docker 就会重新下载约 10 GB 的基础镜像，即使
`isaac-lab-ros2-lqf:latest` 已经存在。

本 README 第 3 节的 `docker compose ... up --no-build` 专门用于复用现有
镜像并启用 X11。

## 9. 常见问题

### 启动很慢或界面卡顿

Isaac Sim/Isaac Lab 第一次启动时会初始化 Kit、编译 RTX shader/PSO，并建立
材质和扩展缓存，可能需要较长时间。第 3 节使用的命名卷会保留这些缓存，后续
启动通常会明显加快。

不要同时启动多个 Kit 实例。`Ctrl+Z` 只会挂起进程，Kit/OmniHub 的锁文件仍
可能保留，后续实例会反复等待 Hub 配置文件或 viewport。退出时使用 `Ctrl+C`
或关闭窗口；排查时可确认当前只有一个 Kit/Hub：

```bash
docker exec isaac-lab-ros2-lqf bash -lc \
  'ps -eo pid,stat,args | grep -E "[k]it|[h]ub|[i]saaclab"'
```

### 执行命令后没有窗口

确认同时满足以下条件：

```bash
echo "$DISPLAY"
docker exec isaac-lab-ros2-lqf bash -lc 'echo "$DISPLAY"'
docker inspect isaac-lab-ros2-lqf | grep /tmp/.X11-unix
```

宿主机和容器内的 `DISPLAY` 都不应为空，并且容器必须有
`/tmp/.X11-unix` 挂载。

### `Could not connect to display`

检查 `XAUTHORITY` 文件是否存在并可读：

```bash
ls -l "$XAUTHORITY"
xauth -f "$XAUTHORITY" list "$DISPLAY"
```

然后重新执行第 3 节的命令。

### `ROS2 Bridge startup failed` 或找不到 `librcutils.so`

在启动 Isaac Sim/Isaac Lab 的同一个 shell 中执行：

```bash
source /opt/ros/jazzy/setup.bash
export LD_LIBRARY_PATH="/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:${LD_LIBRARY_PATH:-}"
```

使用第 5.3 节的宿主机直启命令时，这两行必须放在
`docker exec ... bash -lc` 的命令中，不能只在宿主机 shell 中执行。

### Docker 显示 `unhealthy`

Isaac Lab 基础镜像的默认 healthcheck 检查 Isaac Sim Kit 的 `AppReady`
日志。只运行 ROS 2 或 Gazebo 时没有该日志，`unhealthy` 不代表本 README
中的 Gazebo 控制器启动失败；应以实际窗口和
`ros2 control list_controllers` 的结果为准。

### 旧的 `IsaacLab_launch`

`/home/lqf/script/IsaacLab_launch` 是旧的 Isaac Sim 直启脚本，仍引用已
移除的 `~/docker/isaac-sim` 挂载目录。当前环境不要使用它。
