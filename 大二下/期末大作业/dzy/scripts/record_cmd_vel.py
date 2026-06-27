#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import yaml

import rospy
from geometry_msgs.msg import Twist


class CmdVelRecorder:
    def __init__(self):
        rospy.init_node("cmd_vel_recorder", anonymous=True)

        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.output_file = rospy.get_param("~output_file", self.default_output_file())
        self.min_interval = float(rospy.get_param("~min_interval", 0.0))

        self.records = []
        self.start_time = None
        self.last_record_time = None
        self.last_print_time = 0.0

        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.sub = rospy.Subscriber(
            self.cmd_vel_topic,
            Twist,
            self.cmd_vel_callback,
            queue_size=100
        )

        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("Cmd_vel recorder started.")
        rospy.loginfo("Listening topic: %s", self.cmd_vel_topic)
        rospy.loginfo("Output file: %s", self.output_file)
        rospy.loginfo("Drive the car now. Press Ctrl-C to stop and save.")

    @staticmethod
    def default_output_file():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.expanduser(
            "~/agilex_ws/src/dzy/cmd_vel_paths/cmd_vel_%s.yaml" % stamp
        )

    @staticmethod
    def twist_to_dict(msg):
        return {
            "linear": {
                "x": float(msg.linear.x),
                "y": float(msg.linear.y),
                "z": float(msg.linear.z),
            },
            "angular": {
                "x": float(msg.angular.x),
                "y": float(msg.angular.y),
                "z": float(msg.angular.z),
            },
        }

    def cmd_vel_callback(self, msg):
        now = rospy.Time.now().to_sec()

        if self.start_time is None:
            self.start_time = now

        if self.last_record_time is not None and self.min_interval > 0.0:
            if now - self.last_record_time < self.min_interval:
                return

        rel_time = now - self.start_time
        self.last_record_time = now

        self.records.append({
            "t": round(rel_time, 6),
            "cmd": self.twist_to_dict(msg),
        })

        self.print_status(msg, rel_time)

    def print_status(self, msg, rel_time):
        now = time.time()
        if now - self.last_print_time < 0.5:
            return

        self.last_print_time = now
        sys.stdout.write(
            "\rrecorded=%d  t=%7.2fs  vx=% .3f  wz=% .3f     "
            % (len(self.records), rel_time, msg.linear.x, msg.angular.z)
        )
        sys.stdout.flush()

    def save_to_file(self):
        data = {
            "type": "cmd_vel_path",
            "cmd_vel_topic": self.cmd_vel_topic,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "record_count": len(self.records),
            "duration": self.records[-1]["t"] if self.records else 0.0,
            "records": self.records,
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False
            )

    def on_shutdown(self):
        self.save_to_file()
        print("")
        rospy.loginfo("Saved %d cmd_vel records to: %s",
                      len(self.records), self.output_file)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        CmdVelRecorder().run()
    except rospy.ROSInterruptException:
        pass
