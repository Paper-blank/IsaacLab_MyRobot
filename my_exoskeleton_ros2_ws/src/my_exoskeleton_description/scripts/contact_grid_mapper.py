#!/usr/bin/env python3
"""Map Gazebo contact points into one 10x6 normal-force grid per foot."""

import math

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener


class ContactGridMapper(Node):
    """Convert raw contact points into a fixed-size 60-cell force array."""

    def __init__(self):
        super().__init__("contact_grid_mapper")

        self.rows = 6
        self.cols = 10
        self.world_frame = self.declare_parameter("world_frame", "world").value
        self.left_foot_frame = self.declare_parameter("left_foot_frame", "Lfoot").value
        self.right_foot_frame = self.declare_parameter("right_foot_frame", "Rfoot").value

        # Local bounds of the current Lfoot.STL and Rfoot.STL collision meshes.
        self.bounds = {
            # Include a small collision-mesh margin so contact points on the
            # outer edge are not discarded by floating-point/mesh offsets.
            "left": (-0.10, 0.25, -0.06, 0.13),
            "right": (-0.11, 0.25, -0.125, 0.06),
        }

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.left_pub = self.create_publisher(
            Float64MultiArray, "/left_foot_pressure_grid", 10
        )
        self.right_pub = self.create_publisher(
            Float64MultiArray, "/right_foot_pressure_grid", 10
        )
        self.create_subscription(
            Contacts,
            "/left_foot_contact",
            self.left_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Contacts,
            "/right_foot_contact",
            self.right_callback,
            qos_profile_sensor_data,
        )
        self.latest_left = None
        self.latest_right = None
        self.left_received_at = None
        self.right_received_at = None
        self.contact_timeout_ns = 150_000_000
        self.publish_timer = self.create_timer(1.0 / 20.0, self.publish_grids)

    def _grid_message(self, grid):
        message = Float64MultiArray()
        message.layout.dim = [
            MultiArrayDimension(label="y", size=self.rows, stride=self.rows * self.cols),
            MultiArrayDimension(label="x", size=self.cols, stride=self.cols),
        ]
        message.data = grid
        return message

    def _lookup_transform(self, message, target_frame):
        source_frame = message.header.frame_id or self.world_frame
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.01),
            )
            return transform
        except TransformException as error:
            self.get_logger().warning(
                f"Cannot transform contact point from {source_frame} "
                f"to {target_frame}: {error}",
                throttle_duration_sec=2.0,
            )
            return None

    @staticmethod
    def _normal_force(contact, contact_index):
        if not contact.wrenches:
            return 0.0

        wrench = contact.wrenches[min(contact_index, len(contact.wrenches) - 1)]
        force = wrench.body_1_wrench.force
        if math.isclose(force.x, 0.0) and math.isclose(force.y, 0.0) and math.isclose(force.z, 0.0):
            force = wrench.body_2_wrench.force

        if contact.normals:
            normal = contact.normals[min(contact_index, len(contact.normals) - 1)]
            normal_length = math.sqrt(normal.x**2 + normal.y**2 + normal.z**2)
            if normal_length > 1e-9:
                return max(
                    0.0,
                    abs(force.x * normal.x + force.y * normal.y + force.z * normal.z)
                    / normal_length,
                )

        return math.sqrt(force.x**2 + force.y**2 + force.z**2)

    def _map_contacts(self, message, side, target_frame):
        x_min, x_max, y_min, y_max = self.bounds[side]
        grid = [0.0] * (self.rows * self.cols)
        transform = self._lookup_transform(message, target_frame)
        if transform is None:
            return grid

        for contact in message.contacts:
            for index, position in enumerate(contact.positions):
                point = PointStamped()
                point.header = message.header
                point.header.frame_id = message.header.frame_id or self.world_frame
                point.point = position
                local_position = do_transform_point(point, transform).point
                if not (x_min <= local_position.x <= x_max and y_min <= local_position.y <= y_max):
                    continue

                x_ratio = (local_position.x - x_min) / (x_max - x_min)
                y_ratio = (local_position.y - y_min) / (y_max - y_min)
                x_index = min(self.cols - 1, max(0, int(x_ratio * self.cols)))
                y_index = min(self.rows - 1, max(0, int(y_ratio * self.rows)))
                grid[y_index * self.cols + x_index] += self._normal_force(contact, index)

        return grid

    def publish_grids(self):
        now_ns = self.get_clock().now().nanoseconds
        left_grid = (
            self._map_contacts(self.latest_left, "left", self.left_foot_frame)
            if self.latest_left is not None
            and now_ns - self.left_received_at <= self.contact_timeout_ns
            else [0.0] * (self.rows * self.cols)
        )
        right_grid = (
            self._map_contacts(self.latest_right, "right", self.right_foot_frame)
            if self.latest_right is not None
            and now_ns - self.right_received_at <= self.contact_timeout_ns
            else [0.0] * (self.rows * self.cols)
        )
        self.left_pub.publish(self._grid_message(left_grid))
        self.right_pub.publish(self._grid_message(right_grid))

    def left_callback(self, message):
        self.latest_left = message
        self.left_received_at = self.get_clock().now().nanoseconds

    def right_callback(self, message):
        self.latest_right = message
        self.right_received_at = self.get_clock().now().nanoseconds


def main(args=None):
    rclpy.init(args=args)
    node = ContactGridMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
