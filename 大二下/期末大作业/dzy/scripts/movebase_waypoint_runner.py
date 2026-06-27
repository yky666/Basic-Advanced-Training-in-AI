#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import yaml

import rospy
import actionlib
import tf

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class MoveBaseWaypointRunner:
    def __init__(self):
        rospy.init_node("movebase_waypoint_runner", anonymous=True)

        self.waypoints_file = rospy.get_param(
            "~waypoints_file",
            os.path.expanduser("~/agilex_ws/src/dzy/config/route_a2_b2.yaml")
        )

        # 例如：
        # _sequence:=p001,p002,p003
        # 如果不传 sequence，就默认按 waypoints.yaml 里的顺序全部执行
        self.sequence_param = rospy.get_param("~sequence", "")

        self.goal_timeout = float(rospy.get_param("~goal_timeout", 60.0))
        self.pause_at_each = self.parse_bool(rospy.get_param("~pause_at_each", False))

        # 每到达一个点后，原地等待几秒再前往下一个点
        self.wait_after_goal = float(rospy.get_param("~wait_after_goal", 2.0))

        # 提前判定到达的容差
        # xy_tolerance: 距离目标点多少米以内算到达
        # yaw_tolerance: 朝向误差多少弧度以内算到达
        self.xy_tolerance = float(rospy.get_param("~xy_tolerance", 0.20))
        self.yaw_tolerance = float(rospy.get_param("~yaw_tolerance", 0.35))

        # 小车本体坐标系
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        # 检查到达状态的频率
        self.arrival_check_rate = float(rospy.get_param("~arrival_check_rate", 10.0))

        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.tf_listener = tf.TransformListener()

        rospy.loginfo("Waiting for move_base action server...")
        if not self.client.wait_for_server(rospy.Duration(20.0)):
            rospy.logerr("Cannot connect to move_base action server.")
            raise RuntimeError("move_base action server not available")

        rospy.loginfo("Connected to move_base.")

        # 给 TF 一点缓存时间
        rospy.sleep(0.5)

        self.frame_id, self.waypoints = self.load_waypoints(self.waypoints_file)
        self.sequence = self.build_sequence()

        rospy.loginfo("Waypoints file: %s", self.waypoints_file)
        rospy.loginfo("Frame id: %s", self.frame_id)
        rospy.loginfo("Base frame: %s", self.base_frame)
        rospy.loginfo("XY tolerance: %.3f m", self.xy_tolerance)
        rospy.loginfo(
            "Yaw tolerance: %.3f rad = %.2f deg",
            self.yaw_tolerance,
            math.degrees(self.yaw_tolerance)
        )
        rospy.loginfo("Goal timeout: %.1f s", self.goal_timeout)
        rospy.loginfo("Wait after each goal: %.1f s", self.wait_after_goal)
        rospy.loginfo("Mission sequence: %s", " -> ".join(self.sequence))

    def parse_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ["true", "1", "yes", "y"]
        return bool(value)

    def load_waypoints(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Waypoints file not found: %s" % path)

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise RuntimeError("Waypoints file is empty: %s" % path)

        frame_id = data.get("frame_id", "map")
        waypoint_list = data.get("waypoints", [])

        if not waypoint_list:
            raise RuntimeError("No waypoints found in file: %s" % path)

        waypoints = {}
        for p in waypoint_list:
            name = p.get("name", None)
            if not name:
                rospy.logwarn("Skip waypoint without name: %s", str(p))
                continue

            if name in waypoints:
                rospy.logwarn(
                    "Duplicate waypoint name: %s, later one will overwrite previous one.",
                    name
                )

            waypoints[name] = p

        if not waypoints:
            raise RuntimeError("No valid named waypoints found.")

        return frame_id, waypoints

    def build_sequence(self):
        if self.sequence_param.strip():
            sequence = [s.strip() for s in self.sequence_param.split(",") if s.strip()]
        else:
            # 如果没有指定 sequence，就按 YAML 文件原顺序执行
            with open(self.waypoints_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            sequence = [p["name"] for p in data["waypoints"] if "name" in p]

        missing = [name for name in sequence if name not in self.waypoints]
        if missing:
            raise RuntimeError("These waypoint names are missing in yaml: %s" % missing)

        return sequence

    def make_goal(self, point):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.frame_id
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = float(point["x"])
        goal.target_pose.pose.position.y = float(point["y"])
        goal.target_pose.pose.position.z = float(point.get("z", 0.0))

        # 优先使用采点文件里的 quaternion
        if "orientation" in point:
            q = point["orientation"]
            goal.target_pose.pose.orientation.x = float(q.get("x", 0.0))
            goal.target_pose.pose.orientation.y = float(q.get("y", 0.0))
            goal.target_pose.pose.orientation.z = float(q.get("z", 0.0))
            goal.target_pose.pose.orientation.w = float(q.get("w", 1.0))
        else:
            # 如果没有 orientation，就用 yaw 转四元数
            yaw = float(point.get("yaw", 0.0))
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
            goal.target_pose.pose.orientation.x = qx
            goal.target_pose.pose.orientation.y = qy
            goal.target_pose.pose.orientation.z = qz
            goal.target_pose.pose.orientation.w = qw

        return goal

    def angle_diff(self, a, b):
        """
        返回两个角度之间的最短差值，范围 [-pi, pi]
        """
        d = a - b
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return d

    def get_robot_pose(self):
        """
        获取小车在 frame_id 下的位置。
        通常就是 map -> base_link。
        """
        try:
            trans, rot = self.tf_listener.lookupTransform(
                self.frame_id,
                self.base_frame,
                rospy.Time(0)
            )

            _, _, yaw = euler_from_quaternion(rot)
            return trans[0], trans[1], yaw

        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException
        ) as e:
            rospy.logwarn_throttle(
                1.0,
                "Cannot get robot pose %s -> %s: %s",
                self.frame_id,
                self.base_frame,
                str(e)
            )
            return None

    def get_target_yaw(self, point):
        """
        从 waypoint 中读取目标朝向。
        优先使用 yaw；如果没有 yaw，则从 orientation 四元数中计算。
        """
        if "yaw" in point:
            return float(point["yaw"])

        if "orientation" in point:
            q = point["orientation"]
            quat = [
                float(q.get("x", 0.0)),
                float(q.get("y", 0.0)),
                float(q.get("z", 0.0)),
                float(q.get("w", 1.0)),
            ]
            _, _, yaw = euler_from_quaternion(quat)
            return yaw

        return 0.0

    def is_close_enough(self, point):
        """
        判断是否已经足够接近目标点。

        默认要求：
        1. 距离误差 <= xy_tolerance
        2. 朝向误差 <= yaw_tolerance

        如果某个点在 YAML 里设置了：
        position_only: true

        那么该点只判断距离，不判断朝向。
        """
        pose = self.get_robot_pose()
        if pose is None:
            return False

        rx, ry, ryaw = pose

        tx = float(point["x"])
        ty = float(point["y"])
        tyaw = self.get_target_yaw(point)

        dist = math.hypot(rx - tx, ry - ty)
        yaw_err = abs(self.angle_diff(ryaw, tyaw))

        position_only = self.parse_bool(point.get("position_only", False))

        if position_only:
            rospy.loginfo_throttle(
                1.0,
                "Distance to goal: %.3f m | position_only=True | tolerance=%.3f m",
                dist,
                self.xy_tolerance
            )
            return dist <= self.xy_tolerance

        rospy.loginfo_throttle(
            1.0,
            "Distance to goal: %.3f m | yaw error: %.2f deg | tolerance: %.3f m, %.2f deg",
            dist,
            math.degrees(yaw_err),
            self.xy_tolerance,
            math.degrees(self.yaw_tolerance)
        )

        return dist <= self.xy_tolerance and yaw_err <= self.yaw_tolerance

    def wait_after_reach(self, name):
        """
        到达一个 waypoint 后原地等待一段时间。
        """
        if self.wait_after_goal <= 0:
            return

        rospy.loginfo(
            "Reached waypoint: %s. Waiting %.1f seconds before next goal...",
            name,
            self.wait_after_goal
        )

        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed >= self.wait_after_goal:
                break
            rate.sleep()

    def send_one_goal(self, name):
        point = self.waypoints[name]
        goal = self.make_goal(point)

        rospy.loginfo("")
        rospy.loginfo("Sending goal: %s", name)
        rospy.loginfo(
            "Target: x=%.3f, y=%.3f, yaw_deg=%s",
            float(point["x"]),
            float(point["y"]),
            str(point.get("yaw_deg", "unknown"))
        )

        if self.parse_bool(point.get("position_only", False)):
            rospy.loginfo("This waypoint is position_only=True, yaw will not be checked.")

        self.client.send_goal(goal)

        start_time = rospy.Time.now()
        rate = rospy.Rate(self.arrival_check_rate)

        while not rospy.is_shutdown():
            # 自己判断是否已经足够接近目标点
            if self.is_close_enough(point):
                rospy.loginfo(
                    "Close enough to goal: %s. Cancel current goal.",
                    name
                )
                self.client.cancel_goal()
                rospy.sleep(0.5)
                self.wait_after_reach(name)
                return True

            # 如果 move_base 自己已经判定成功，也继续
            state = self.client.get_state()

            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("Goal reached by move_base: %s", name)
                self.wait_after_reach(name)
                return True

            if state in [
                GoalStatus.ABORTED,
                GoalStatus.REJECTED,
                GoalStatus.LOST
            ]:
                rospy.logwarn("Goal failed: %s, state=%d", name, state)
                return False

            # 如果被外部取消，通常也认为任务中断
            if state == GoalStatus.PREEMPTED:
                rospy.logwarn("Goal preempted externally: %s, state=%d", name, state)
                return False

            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > self.goal_timeout:
                rospy.logwarn("Goal timeout: %s. Canceling goal.", name)
                self.client.cancel_goal()
                rospy.sleep(1.0)
                return False

            rate.sleep()

        return False

    def run(self):
        rospy.loginfo("Start running waypoints.")

        for i, name in enumerate(self.sequence):
            if rospy.is_shutdown():
                break

            if self.pause_at_each:
                input("\nPress Enter to send next goal: %s ..." % name)

            ok = self.send_one_goal(name)

            if not ok:
                rospy.logwarn("Mission stopped at waypoint: %s", name)
                rospy.logwarn("You can check RViz / move_base status and retry.")
                return

        rospy.loginfo("")
        rospy.loginfo("All waypoints finished.")

    def cancel_current_goal(self):
        rospy.logwarn("Cancel current move_base goal.")
        self.client.cancel_goal()


if __name__ == "__main__":
    runner = None

    try:
        runner = MoveBaseWaypointRunner()
        runner.run()

    except KeyboardInterrupt:
        if runner is not None:
            runner.cancel_current_goal()
        print("")
        rospy.logwarn("Interrupted by user.")

    except Exception as e:
        rospy.logerr("Error: %s", str(e))