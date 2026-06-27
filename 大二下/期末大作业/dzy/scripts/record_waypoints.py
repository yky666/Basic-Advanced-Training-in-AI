#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import yaml
import select
import termios
import tty

import rospy
import tf
from tf.transformations import euler_from_quaternion


class WaypointRecorder:
    def __init__(self):
        rospy.init_node("waypoint_recorder", anonymous=True)

        self.target_frame = rospy.get_param("~target_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.output_file = rospy.get_param(
            "~output_file",
            os.path.expanduser("~/agilex_ws/src/dzy/waypoints.yaml")
        )

        self.listener = tf.TransformListener()
        self.waypoints = []
        self.counter = 1

        self.last_print_time = 0.0

        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        rospy.loginfo("Waypoint recorder started.")
        rospy.loginfo("Listening TF: %s -> %s", self.target_frame, self.base_frame)
        rospy.loginfo("Output file: %s", self.output_file)
        rospy.loginfo("Keys: [SPACE] save auto point | [n] save named point | [q] quit")

    def get_current_pose(self):
        """
        获取 base_frame 在 target_frame 下的位姿。
        返回：x, y, yaw, quaternion
        """
        try:
            trans, rot = self.listener.lookupTransform(
                self.target_frame,
                self.base_frame,
                rospy.Time(0)
            )

            x = trans[0]
            y = trans[1]
            z = trans[2]

            roll, pitch, yaw = euler_from_quaternion(rot)

            return {
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw,
                "yaw_deg": math.degrees(yaw),
                "qx": rot[0],
                "qy": rot[1],
                "qz": rot[2],
                "qw": rot[3],
            }

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "Cannot get TF %s -> %s: %s",
                                   self.target_frame, self.base_frame, str(e))
            return None

    def save_to_file(self):
        data = {
            "frame_id": self.target_frame,
            "base_frame": self.base_frame,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "waypoints": self.waypoints
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False
            )

    def record_point(self, name=None):
        pose = self.get_current_pose()
        if pose is None:
            rospy.logwarn("No valid pose. Point not recorded.")
            return

        if name is None or name.strip() == "":
            name = "p%03d" % self.counter

        point = {
            "name": name,
            "x": round(pose["x"], 4),
            "y": round(pose["y"], 4),
            "z": round(pose["z"], 4),
            "yaw": round(pose["yaw"], 6),
            "yaw_deg": round(pose["yaw_deg"], 3),
            "orientation": {
                "x": round(pose["qx"], 6),
                "y": round(pose["qy"], 6),
                "z": round(pose["qz"], 6),
                "w": round(pose["qw"], 6),
            },
            "stamp": rospy.Time.now().to_sec()
        }

        self.waypoints.append(point)
        self.counter += 1
        self.save_to_file()

        rospy.loginfo(
            "Saved %-16s x=%.4f, y=%.4f, yaw=%.3f deg",
            point["name"], point["x"], point["y"], point["yaw_deg"]
        )

    def print_current_pose(self):
        now = time.time()
        if now - self.last_print_time < 1.0:
            return

        self.last_print_time = now
        pose = self.get_current_pose()
        if pose is None:
            return

        sys.stdout.write(
            "\rCurrent pose in %-6s | x=%7.3f  y=%7.3f  yaw=%7.2f deg | saved=%d     "
            % (
                self.target_frame,
                pose["x"],
                pose["y"],
                pose["yaw_deg"],
                len(self.waypoints)
            )
        )
        sys.stdout.flush()

    def run(self):
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())
            rate = rospy.Rate(20)

            while not rospy.is_shutdown():
                self.print_current_pose()

                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)

                    if key == " ":
                        print("")
                        self.record_point()

                    elif key == "n":
                        print("")
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        name = input("Input waypoint name: ").strip()
                        tty.setcbreak(sys.stdin.fileno())
                        self.record_point(name)

                    elif key == "q":
                        print("")
                        rospy.loginfo("Quit. Total saved waypoints: %d", len(self.waypoints))
                        break

                rate.sleep()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.save_to_file()
            print("")
            rospy.loginfo("Waypoints saved to: %s", self.output_file)


if __name__ == "__main__":
    try:
        recorder = WaypointRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        pass