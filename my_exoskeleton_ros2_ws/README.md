# my_exoskeleton ROS 2 workspace

This workspace is mounted into the Isaac Lab ROS 2 container at
`/workspace/ros2_ws`.

The canonical URDF and meshes remain in the Isaac Lab checkout:

`/workspace/isaaclab/source/isaaclab_assets/data/my_robot`

The package copies those assets into its install space during `colcon build`,
and rewrites only the installed URDF's controller path.

## Start the container

For a normal restart, use the existing ROS 2 container directly:

```bash
docker start isaac-lab-ros2-lqf
docker exec -it isaac-lab-ros2-lqf bash
```

The host workspace is mounted at `/workspace/ros2_ws`. Isaac Lab source assets
are mounted separately at `/workspace/isaaclab/source`, so changing the
canonical URDF or meshes requires rebuilding this workspace before launching.

## Build in the Isaac Lab container

```bash
source /opt/ros/jazzy/setup.bash
cd /workspace/ros2_ws
colcon build --symlink-install \
  --cmake-args \
  -DISAACLAB_ASSET_DIR=/workspace/isaaclab/source/isaaclab_assets/data/my_robot
source install/setup.bash
```

## Launch Gazebo Harmonic

The default launch is server-only and does not need X11:

```bash
ros2 launch my_exoskeleton_description gazebo.launch.py
```

To launch the Gazebo GUI, start the container with Isaac Lab X11 forwarding
enabled and then use:

```bash
ros2 launch my_exoskeleton_description gazebo.launch.py headless:=false
```

Check the controller state with:

```bash
ros2 control list_controllers -c /controller_manager
```

The package uses Gazebo Harmonic, `gz_ros2_control`, and bridges Gazebo time
to ROS 2 on `/clock`.
