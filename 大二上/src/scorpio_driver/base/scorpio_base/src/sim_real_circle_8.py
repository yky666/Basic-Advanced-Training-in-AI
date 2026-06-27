#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
from geometry_msgs.msg import Twist

class SimpleFigureEightController:
    def __init__(self):
        rospy.init_node('simple_figure_eight_controller', anonymous=True)
        
        # 固定参数设置（根据题目要求）,线速度linear_speed（任务二）, 圆半径circle_radius（任务二）
        ###########需补全##############

        self.circle_radius=0.5
        self.linear_speed=0.2

      # 计算运动参数：角速度angular_speed（任务二）, 周期（任务二、三）
        ###########需补全##############
        self.angular_speed=0.4
        self.period=(math.pi*self.circle_radius/self.linear_speed)*2

        self.start_time=rospy.get_time()
        self.current_angular=self.angular_speed


        # 创建速度发布器（任务二）
        ###########需补全##############
        self.cmd_vel_pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10) 

        
  
        
        # 运动状态phase（任务三）
        ###########需补全##############
        

        
        # 创建控制定时器（10Hz控制频率）
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("简易版8字轨迹控制器初始化完成")
        # 在终端打印小车线速度，角速度，周期信息（任务二）
        ###########需补全##############
        # 打印小车参数信息
        rospy.loginfo("线速度: %.2f m/s", self.linear_speed)
        rospy.loginfo("角速度: %.2f rad/s", self.angular_speed)
        rospy.loginfo("周期: %.2f s", self.period)

    def control_loop(self, event):
        """主控制循环"""
        try:
            # 检查是否到达切换时间, 如果到达切换时间切换角速度符号
            # 当到达切换时间时，在终端打印提示信息，例如“正在切换运动方向”（任务三）
            ###########需补全##############
            # 计算当前运行时间
            current_time = rospy.get_time()
            elapsed_time = current_time - self.start_time
            
            # 检查是否到达切换时间（每个周期切换一次方向）
            if elapsed_time % (2 * self.period) > self.period:
                new_angular = -self.angular_speed
            else:
                new_angular = self.angular_speed
                
            # 如果角速度发生变化，打印提示信息
            if new_angular != self.current_angular:
                self.current_angular = new_angular
                rospy.loginfo("正在切换运动方向")
            
            # 创建并发布速度指令（任务二、三）
            ###########需补全##############
            # 创建速度指令
            twist = Twist()
            twist.linear.x = self.linear_speed  # 设置线速度（前进方向）
            twist.angular.z = self.current_angular  # 设置角速度（转向）
            
            # 发布速度指令
            self.cmd_vel_pub.publish(twist)
            
        except Exception as e:
            rospy.logerr("控制循环错误: %s", str(e))

    def run(self):
        """主运行函数"""
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = SimpleFigureEightController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序结束")