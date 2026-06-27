#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import select
import termios
import tty
from geometry_msgs.msg import Twist


class LimoKeyboardControl:
    def __init__(self):
        rospy.init_node("limo_keyboard_control", anonymous=True)

        # 可在 launch 文件中修改的默认速度参数
        self.linear_speed = rospy.get_param("~linear_speed", 0.20)
        self.angular_speed = rospy.get_param("~angular_speed", 0.80)

        # 每次按键调速的增量
        self.linear_step = rospy.get_param("~linear_step", 0.05)
        self.angular_step = rospy.get_param("~angular_step", 0.10)

        # 最大速度限制，防止误操作过快
        self.max_linear_speed = rospy.get_param("~max_linear_speed", 1.00)
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 2.00)

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.settings = termios.tcgetattr(sys.stdin)

        self.print_help()

    def print_help(self):
        print("")
        print("========== LIMO 自定义键盘控制节点 ==========")
        print("控制按键：")
        print("    w    前进")
        print("    s    后退")
        print("    a    左转")
        print("    d    右转")
        print("    q    左前转")
        print("    e    右前转")
        print("    z    左后转")
        print("    c    右后转")
        print("    空格 停止")
        print("")
        print("速度调节：")
        print("    i    增大线速度")
        print("    k    减小线速度")
        print("    j    增大角速度")
        print("    l    减小角速度")
        print("")
        print("退出：")
        print("    Ctrl + C")
        print("")
        print("当前默认线速度: %.2f m/s" % self.linear_speed)
        print("当前默认角速度: %.2f rad/s" % self.angular_speed)
        print("===========================================")
        print("")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)

        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ""

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def limit_speed(self):
        if self.linear_speed > self.max_linear_speed:
            self.linear_speed = self.max_linear_speed
        if self.linear_speed < 0.0:
            self.linear_speed = 0.0

        if self.angular_speed > self.max_angular_speed:
            self.angular_speed = self.max_angular_speed
        if self.angular_speed < 0.0:
            self.angular_speed = 0.0

    def publish_cmd(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = angular_z

        self.cmd_pub.publish(twist)

    def run(self):
        try:
            while not rospy.is_shutdown():
                key = self.get_key()

                linear_x = 0.0
                angular_z = 0.0

                if key == "w":
                    linear_x = self.linear_speed
                    angular_z = 0.0

                elif key == "s":
                    linear_x = -self.linear_speed
                    angular_z = 0.0

                elif key == "a":
                    linear_x = 0.0
                    angular_z = self.angular_speed

                elif key == "d":
                    linear_x = 0.0
                    angular_z = -self.angular_speed

                elif key == "q":
                    linear_x = self.linear_speed
                    angular_z = self.angular_speed

                elif key == "e":
                    linear_x = self.linear_speed
                    angular_z = -self.angular_speed

                elif key == "z":
                    linear_x = -self.linear_speed
                    angular_z = -self.angular_speed

                elif key == "c":
                    linear_x = -self.linear_speed
                    angular_z = self.angular_speed

                elif key == " ":
                    linear_x = 0.0
                    angular_z = 0.0

                elif key == "i":
                    self.linear_speed += self.linear_step
                    self.limit_speed()
                    print("线速度增加到: %.2f m/s" % self.linear_speed)
                    continue

                elif key == "k":
                    self.linear_speed -= self.linear_step
                    self.limit_speed()
                    print("线速度减小到: %.2f m/s" % self.linear_speed)
                    continue

                elif key == "j":
                    self.angular_speed += self.angular_step
                    self.limit_speed()
                    print("角速度增加到: %.2f rad/s" % self.angular_speed)
                    continue

                elif key == "l":
                    self.angular_speed -= self.angular_step
                    self.limit_speed()
                    print("角速度减小到: %.2f rad/s" % self.angular_speed)
                    continue

                elif key == "\x03":
                    break

                else:
                    # 没有按键时持续发布停止，防止小车失控
                    linear_x = 0.0
                    angular_z = 0.0

                self.publish_cmd(linear_x, angular_z)

        except Exception as e:
            print(e)

        finally:
            self.publish_cmd(0.0, 0.0)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


if __name__ == "__main__":
    controller = LimoKeyboardControl()
    controller.run()
