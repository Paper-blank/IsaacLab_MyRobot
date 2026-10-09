#!/usr/bin/env python3
"""Publish the latest Gazebo sensor samples at the requested ROS rates."""

from collections import OrderedDict

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import Imu


class SensorRateLimiter(Node):
    """Keep the public sensor topics at 200 Hz (IMU) and 20 Hz (contact)."""

    IMU_TOPICS = (
        "rthigh",
        "rleg",
        "rfoot",
        "lthigh",
        "lleg",
        "lfoot",
    )

    def __init__(self):
        super().__init__("sensor_rate_limiter")

        self.latest_imu = OrderedDict((name, None) for name in self.IMU_TOPICS)
        self.imu_publishers = {
            name: self.create_publisher(
                Imu, f"/my_exoskeleton/imu/{name}", qos_profile_sensor_data
            )
            for name in self.IMU_TOPICS
        }
        for name in self.IMU_TOPICS:
            self.create_subscription(
                Imu,
                f"/my_exoskeleton/imu/{name}_raw",
                lambda message, sensor_name=name: self._save_imu(sensor_name, message),
                qos_profile_sensor_data,
            )

        self.latest_contacts = {"left": None, "right": None}
        self.contact_publishers = {
            "left": self.create_publisher(
                Contacts, "/left_foot_contact", qos_profile_sensor_data
            ),
            "right": self.create_publisher(
                Contacts, "/right_foot_contact", qos_profile_sensor_data
            ),
        }
        for side in ("left", "right"):
            self.create_subscription(
                Contacts,
                f"/{side}_foot_contact_raw",
                lambda message, foot_side=side: self._save_contact(foot_side, message),
                qos_profile_sensor_data,
            )

        self.create_timer(1.0 / 200.0, self._publish_imu)
        self.create_timer(1.0 / 20.0, self._publish_contacts)

    def _save_imu(self, name, message):
        self.latest_imu[name] = message

    def _save_contact(self, side, message):
        self.latest_contacts[side] = message

    def _publish_imu(self):
        for name, message in self.latest_imu.items():
            if message is not None:
                self.imu_publishers[name].publish(message)

    def _publish_contacts(self):
        for side, message in self.latest_contacts.items():
            if message is not None:
                self.contact_publishers[side].publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = SensorRateLimiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
