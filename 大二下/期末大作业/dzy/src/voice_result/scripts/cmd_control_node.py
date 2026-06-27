#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class EnhancedCmdControl:
    def __init__(self):
        self.linear_speed = rospy.get_param('~linear_speed', 0.2)
        self.min_turn_speed = rospy.get_param('~min_turn_speed', 0.25)
        self.angular_speed = rospy.get_param('~angular_speed', 0.35)
        self.publish_rate = rospy.get_param('~publish_rate', 20)

        self.intent_velocity_map = {
            'forward':     (self.linear_speed, 0.0),
            'backward':    (-self.linear_speed, 0.0),
            'left':        (0.0, self.angular_speed),
            'right':       (0.0, -self.angular_speed),
            'stop':        (0.0, 0.0),
            'speed_up':    (self.linear_speed * 1.5, 0.0),
            'slow_down':   (self.linear_speed * 0.5, 0.0),
            'turn_around': (self.linear_speed * 0.5, self.angular_speed * 2)
        }

        self.current_twist = Twist()
        self.publish_timer = None
        self.stop_timer = None

        rospy.Subscriber('/cmd_params', String, self.cmd_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.status_pub = rospy.Publisher('/robot_status', String, queue_size=10)
        rospy.loginfo(f"✅ 控制节点启动 (linear={self.linear_speed}, angular={self.angular_speed})")

    # ==========================
    # 🔴 核心修复：每次新指令，彻底清空所有历史定时器！
    # ==========================
    def _clear_all_timers(self):
        # 清空发送定时器
        if self.publish_timer is not None:
            self.publish_timer.shutdown()
            self.publish_timer = None
        # 清空停止定时器
        if self.stop_timer is not None:
            self.stop_timer.shutdown()
            self.stop_timer = None

    def cmd_callback(self, msg):
        cmd_str = msg.data
        rospy.loginfo(f"📥 收到指令: {cmd_str}")

        # ==========================
        # 🔴 来新指令第一件事：杀死所有旧的定时器！
        # ==========================
        self._clear_all_timers()

        intent, params = self._parse_command(cmd_str)
        linear, angular = self._calculate_twist(intent, params)

        # 设置当前速度
        self.current_twist.linear.x = linear
        self.current_twist.angular.z = angular

        # 立即发布一次
        self.cmd_vel_pub.publish(self.current_twist)
        self.status_pub.publish(f"执行: {intent}")

        # 启动持续发送
        self.publish_timer = rospy.Timer(
            rospy.Duration(1.0/self.publish_rate),
            self._publish_cb,
            oneshot=False
        )

        # 计算自动停止时间
        duration = self._calculate_duration(intent, params)
        if duration > 0:
            self.stop_timer = rospy.Timer(
                rospy.Duration(duration),
                self._stop_cb,
                oneshot=True
            )
            rospy.loginfo(f"⏱️  {duration:.1f} 秒后自动停止")

    def _publish_cb(self, event):
        self.cmd_vel_pub.publish(self.current_twist)

    # ==========================
    # 🔴 修复：停止时也清空所有定时器
    # ==========================
    def _stop_cb(self, event):
        self._clear_all_timers()
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        self.status_pub.publish("已停止")
        rospy.loginfo("✅ 机器人已停止")

    def _calculate_twist(self, intent, params):
        linear, angular = self.intent_velocity_map.get(intent, (0.0, 0.0))
        if intent in ['left', 'right'] and linear == 0:
            linear = self.min_turn_speed
        return linear, angular

    def _calculate_duration(self, intent, params):
        if 'distance' in params and intent in ['forward','backward']:
            return abs(params['distance']) / self.linear_speed
        elif 'angle' in params and intent in ['left','right','turn_around']:
            return (abs(params['angle']) * 3.14159 / 180.0) / self.angular_speed
        elif 'time' in params:
            return params['time']
        return 0.0

    def _parse_command(self, cmd_str):
        parts = cmd_str.split(',')
        intent = parts[0].strip()
        params = {}
        for p in parts[1:]:
            if '=' in p:
                k, v = p.split('=')
                try:
                    params[k.strip()] = float(v.strip())
                except:
                    pass
        return intent, params

def main():
    rospy.init_node('cmd_control_node')
    EnhancedCmdControl()
    rospy.spin()

if __name__ == '__main__':
    main()