#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy

from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist


class LineControllerNode:
    def __init__(self):
        rospy.init_node("line_controller_node")

        # -----------------------------
        # ROS 参数
        # -----------------------------
        self.error_topic = rospy.get_param("~error_topic", "/line_follow/error")
        self.valid_topic = rospy.get_param("~valid_topic", "/line_follow/valid")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.done_topic = rospy.get_param("~done_topic", "/line_follow/done")

        # 前进速度
        self.base_speed = rospy.get_param("~base_speed", 0.20)
        self.min_speed = rospy.get_param("~min_speed", 0.06)

        # 转向比例系数
        self.kp = rospy.get_param("~kp", 0.8)

        # 最大角速度限制
        self.max_angular_z = rospy.get_param("~max_angular_z", 1.5)

        # 弯道降速系数：error 越大，速度越低
        self.slowdown_gain = rospy.get_param("~slowdown_gain", 0.45)

        # error 正负号到 angular.z 的映射
        # error > 0 表示目标线在画面右侧
        # angular.z > 0 通常表示左转
        # 因此默认 error > 0 时右转：turn_sign = -1.0
        self.turn_sign = rospy.get_param("~turn_sign", -1.0)

        # 如果超过该时间没收到视觉结果，则认为视觉信息过期
        self.lost_timeout = rospy.get_param("~lost_timeout", 0.5)

        # 控制频率
        self.control_rate = rospy.get_param("~control_rate", 20.0)

        # -----------------------------
        # 巡线完成判定参数
        # -----------------------------
        # 连续多少帧 valid=False 后认为线路结束
        self.no_line_frames_limit = rospy.get_param("~no_line_frames_limit", 20)

        # 启动后至少巡线多少秒，才允许根据“连续丢线”判定完成
        self.min_follow_time = rospy.get_param("~min_follow_time", 6.0)

        # 最大巡线时间，超过后强制认为完成，避免卡死
        self.max_follow_time = rospy.get_param("~max_follow_time", 35.0)

        # done 后是否持续发布零速度
        self.keep_stopping_after_done = rospy.get_param("~keep_stopping_after_done", True)

        # -----------------------------
        # 状态变量
        # -----------------------------
        self.latest_error = 0.0
        self.latest_valid = False

        self.last_error_time = rospy.Time(0)
        self.last_valid_time = rospy.Time(0)

        self.start_time = rospy.Time.now()
        self.no_line_frames = 0
        self.done = False
        self.done_published = False

        # -----------------------------
        # ROS 通信
        # -----------------------------
        self.error_sub = rospy.Subscriber(
            self.error_topic,
            Float32,
            self.error_callback,
            queue_size=10
        )

        self.valid_sub = rospy.Subscriber(
            self.valid_topic,
            Bool,
            self.valid_callback,
            queue_size=10
        )

        self.cmd_pub = rospy.Publisher(
            self.cmd_vel_topic,
            Twist,
            queue_size=10
        )

        # latch=True：即使状态机稍晚订阅，也能收到最近一次 done=True
        self.done_pub = rospy.Publisher(
            self.done_topic,
            Bool,
            queue_size=1,
            latch=True
        )

        rospy.loginfo("line_controller_node started.")
        rospy.loginfo("Subscribing error topic: %s", self.error_topic)
        rospy.loginfo("Subscribing valid topic: %s", self.valid_topic)
        rospy.loginfo("Publishing cmd_vel topic: %s", self.cmd_vel_topic)
        rospy.loginfo("Publishing done topic: %s", self.done_topic)
        rospy.loginfo(
            "Finish condition: no_line_frames_limit=%d, min_follow_time=%.1f, max_follow_time=%.1f",
            self.no_line_frames_limit,
            self.min_follow_time,
            self.max_follow_time
        )

    def error_callback(self, msg):
        self.latest_error = msg.data
        self.last_error_time = rospy.Time.now()

    def valid_callback(self, msg):
        self.latest_valid = msg.data
        self.last_valid_time = rospy.Time.now()

        if self.latest_valid:
            self.no_line_frames = 0
        else:
            self.no_line_frames += 1

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def publish_done_once(self, reason):
        if self.done_published:
            return

        self.done = True
        self.done_published = True

        self.stop_robot()
        self.done_pub.publish(Bool(data=True))

        rospy.logwarn("Line follow done. reason=%s", reason)

    def get_elapsed_time(self):
        return (rospy.Time.now() - self.start_time).to_sec()

    def check_finish_condition(self, error_fresh, valid_fresh):
        elapsed = self.get_elapsed_time()

        # 兜底：超过最大巡线时间，强制结束
        if elapsed >= self.max_follow_time:
            self.publish_done_once("max_follow_time_reached")
            return True

        # 刚启动时不允许因为丢线直接结束，避免前几帧不稳定误判
        if elapsed < self.min_follow_time:
            return False

        # 正常结束条件：连续多帧没有识别到线路
        if self.no_line_frames >= self.no_line_frames_limit:
            self.publish_done_once("no_line_frames_limit_reached")
            return True

        # 如果 detector 不再发布 valid，也可以视为丢线异常；
        # 这里不立即结束，而是交给 max_follow_time 兜底，避免误判。
        if not valid_fresh:
            rospy.logwarn_throttle(
                1.0,
                "valid topic is stale. elapsed=%.1f, wait max_follow_time fallback.",
                elapsed
            )

        return False

    def spin(self):
        rate = rospy.Rate(self.control_rate)

        while not rospy.is_shutdown():
            now = rospy.Time.now()

            error_fresh = (now - self.last_error_time).to_sec() < self.lost_timeout
            valid_fresh = (now - self.last_valid_time).to_sec() < self.lost_timeout

            # 已完成后保持停车，等待状态机关闭 launch
            if self.done:
                if self.keep_stopping_after_done:
                    self.stop_robot()
                rate.sleep()
                continue

            # 检查是否满足巡线完成条件
            if self.check_finish_condition(error_fresh, valid_fresh):
                rate.sleep()
                continue

            # 视觉信息无效或当前帧没识别到线：停车，不继续按旧误差走
            if not error_fresh or not valid_fresh or not self.latest_valid:
                self.stop_robot()

                rospy.loginfo_throttle(
                    0.5,
                    "line invalid or stale. valid=%s no_line_frames=%d elapsed=%.1f",
                    self.latest_valid,
                    self.no_line_frames,
                    self.get_elapsed_time()
                )

                rate.sleep()
                continue

            # 归一化误差限制到 [-1, 1]
            error = self.clamp(self.latest_error, -1.0, 1.0)

            # 转向控制
            angular_z = self.turn_sign * self.kp * error
            angular_z = self.clamp(
                angular_z,
                -self.max_angular_z,
                self.max_angular_z
            )

            # 弯道降速：偏差越大，速度越低
            speed_scale = 1.0 - self.slowdown_gain * abs(error)
            speed_scale = self.clamp(speed_scale, 0.0, 1.0)

            linear_x = self.base_speed * speed_scale
            linear_x = max(linear_x, self.min_speed)

            cmd = Twist()
            cmd.linear.x = linear_x
            cmd.angular.z = angular_z

            self.cmd_pub.publish(cmd)

            rospy.loginfo_throttle(
                0.5,
                "valid=%s error=%.3f linear.x=%.3f angular.z=%.3f no_line_frames=%d elapsed=%.1f",
                self.latest_valid,
                error,
                linear_x,
                angular_z,
                self.no_line_frames,
                self.get_elapsed_time()
            )

            rate.sleep()


if __name__ == "__main__":
    node = LineControllerNode()
    node.spin()