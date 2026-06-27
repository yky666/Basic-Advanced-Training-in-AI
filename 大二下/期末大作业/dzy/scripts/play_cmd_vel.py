#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml

import rospy
from geometry_msgs.msg import Twist


class CmdVelPlayer:
    def __init__(self):
        rospy.init_node("cmd_vel_player", anonymous=True)

        self.input_file = rospy.get_param("~input_file", "")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.speed_scale = float(rospy.get_param("~speed_scale", 1.0))
        self.loop = bool(rospy.get_param("~loop", False))
        self.start_delay = float(rospy.get_param("~start_delay", 0.0))
        self.stop_after_play = bool(rospy.get_param("~stop_after_play", True))

        if not self.input_file:
            raise rospy.ROSException("Missing required param: ~input_file")
        if not os.path.exists(self.input_file):
            raise rospy.ROSException("File not found: %s" % self.input_file)
        if self.speed_scale <= 0.0:
            raise rospy.ROSException("~speed_scale must be greater than 0")

        self.records = self.load_records(self.input_file)
        if not self.records:
            raise rospy.ROSException("No cmd_vel records in: %s" % self.input_file)

        self.pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        rospy.on_shutdown(self.publish_zero)

        rospy.loginfo("Cmd_vel player ready.")
        rospy.loginfo("Input file: %s", self.input_file)
        rospy.loginfo("Publishing topic: %s", self.cmd_vel_topic)
        rospy.loginfo("Records: %d, duration: %.2fs, speed_scale: %.2f",
                      len(self.records), self.records[-1]["t"], self.speed_scale)

    @staticmethod
    def load_records(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("records", [])

    @staticmethod
    def dict_to_twist(item):
        cmd = item.get("cmd", {})
        linear = cmd.get("linear", {})
        angular = cmd.get("angular", {})

        msg = Twist()
        msg.linear.x = float(linear.get("x", 0.0))
        msg.linear.y = float(linear.get("y", 0.0))
        msg.linear.z = float(linear.get("z", 0.0))
        msg.angular.x = float(angular.get("x", 0.0))
        msg.angular.y = float(angular.get("y", 0.0))
        msg.angular.z = float(angular.get("z", 0.0))
        return msg

    def publish_zero(self):
        if hasattr(self, "pub"):
            self.pub.publish(Twist())

    def play_once(self):
        if self.start_delay > 0.0:
            rospy.sleep(self.start_delay)

        last_t = 0.0
        for item in self.records:
            if rospy.is_shutdown():
                break

            t = float(item.get("t", 0.0))
            sleep_time = max(0.0, (t - last_t) / self.speed_scale)
            if sleep_time > 0.0:
                rospy.sleep(sleep_time)

            self.pub.publish(self.dict_to_twist(item))
            last_t = t

        if self.stop_after_play:
            self.publish_zero()

    def run(self):
        rospy.sleep(0.5)
        while not rospy.is_shutdown():
            self.play_once()
            if not self.loop:
                break


if __name__ == "__main__":
    try:
        CmdVelPlayer().run()
    except rospy.ROSInterruptException:
        pass
