import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("my_exoskeleton_description")
    world_path = os.path.join(package_share, "worlds", "empty.sdf")
    urdf_path = os.path.join(package_share, "urdf", "my_exoskeleton.urdf")
    gz_ros2_control_library_path = os.path.join(get_package_prefix("gz_ros2_control"), "lib")
    gz_sim_plugin_path = "/opt/ros/jazzy/opt/gz_sim_vendor/lib/gz-sim-8/plugins"
    gz_physics_engine_path = "/opt/ros/jazzy/opt/gz_physics_vendor/lib/gz-physics-7/engine-plugins"
    gz_resource_path = os.pathsep.join(
        filter(
            None,
            [
                os.path.join(package_share, "urdf"),
                os.path.join(package_share, "models"),
                package_share,
                os.environ.get("GZ_SIM_RESOURCE_PATH"),
            ],
        )
    )
    gz_system_plugin_path = os.pathsep.join(
        filter(
            None,
            [
                gz_ros2_control_library_path,
                gz_sim_plugin_path,
                os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH"),
            ],
        )
    )
    headless = LaunchConfiguration("headless")
    base_z = LaunchConfiguration("base_z")

    with open(urdf_path, "r", encoding="utf-8") as urdf_file:
        robot_description = urdf_file.read()

    gazebo_server = ExecuteProcess(
        cmd=[
            "gz",
            "sim",
            "-r",
            "-s",
            "--physics-engine",
            "gz-physics-bullet-featherstone-plugin",
            world_path,
        ],
        additional_env={
            "GZ_SIM_SYSTEM_PLUGIN_PATH": gz_system_plugin_path,
            "GZ_SIM_PHYSICS_ENGINE_PATH": gz_physics_engine_path,
            "GZ_SIM_RESOURCE_PATH": gz_resource_path,
        },
        condition=IfCondition(headless),
        output="screen",
    )

    gazebo_gui = ExecuteProcess(
        cmd=[
            "gz",
            "sim",
            "-r",
            "--physics-engine",
            "gz-physics-bullet-featherstone-plugin",
            world_path,
        ],
        additional_env={
            "GZ_SIM_SYSTEM_PLUGIN_PATH": gz_system_plugin_path,
            "GZ_SIM_PHYSICS_ENGINE_PATH": gz_physics_engine_path,
            "GZ_SIM_RESOURCE_PATH": gz_resource_path,
        },
        condition=UnlessCondition(headless),
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
        output="screen",
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    imu_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/my_exoskeleton/imu/rthigh@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "/my_exoskeleton/imu/rleg@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "/my_exoskeleton/imu/rfoot@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "/my_exoskeleton/imu/lthigh@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "/my_exoskeleton/imu/lleg@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "/my_exoskeleton/imu/lfoot@sensor_msgs/msg/Imu[gz.msgs.IMU",
        ],
        remappings=[
            ("/my_exoskeleton/imu/rthigh", "/my_exoskeleton/imu/rthigh_raw"),
            ("/my_exoskeleton/imu/rleg", "/my_exoskeleton/imu/rleg_raw"),
            ("/my_exoskeleton/imu/rfoot", "/my_exoskeleton/imu/rfoot_raw"),
            ("/my_exoskeleton/imu/lthigh", "/my_exoskeleton/imu/lthigh_raw"),
            ("/my_exoskeleton/imu/lleg", "/my_exoskeleton/imu/lleg_raw"),
            ("/my_exoskeleton/imu/lfoot", "/my_exoskeleton/imu/lfoot_raw"),
        ],
        output="screen",
    )

    contact_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/left_foot_contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts",
            "/right_foot_contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts",
        ],
        remappings=[
            ("/left_foot_contact", "/left_foot_contact_raw"),
            ("/right_foot_contact", "/right_foot_contact_raw"),
        ],
        output="screen",
    )

    sensor_rate_limiter = Node(
        package="my_exoskeleton_description",
        executable="sensor_rate_limiter.py",
        output="screen",
    )

    # The current robot settles on the ground with base_link near z=0. This
    # reference can be overridden when launching a scenario with another base
    # height; a moving floating base requires a dynamic TF source.
    world_to_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0",
            "--y", "0",
            "--z", base_z,
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "world",
            "--child-frame-id", "base_link",
        ],
        output="screen",
    )

    contact_grid_mapper = Node(
        package="my_exoskeleton_description",
        executable="contact_grid_mapper.py",
        parameters=[
            {"world_frame": "world"},
            {"left_foot_frame": "Lfoot"},
            {"right_foot_frame": "Rfoot"},
        ],
        output="screen",
    )

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "30",
        ],
        output="screen",
    )

    position_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "exoskeleton_position_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "30",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="true",
                description="Run only the Gazebo server. Set false when X11 forwarding is configured.",
            ),
            DeclareLaunchArgument(
                "base_z",
                default_value="0.0",
                description="World height of the fixed base_link reference used by the pressure mapper.",
            ),
            gazebo_server,
            gazebo_gui,
            clock_bridge,
            imu_bridge,
            contact_bridge,
            sensor_rate_limiter,
            world_to_base,
            contact_grid_mapper,
            robot_state_publisher,
            TimerAction(period=6.0, actions=[joint_state_broadcaster]),
            TimerAction(period=8.0, actions=[position_controller]),
        ]
    )
