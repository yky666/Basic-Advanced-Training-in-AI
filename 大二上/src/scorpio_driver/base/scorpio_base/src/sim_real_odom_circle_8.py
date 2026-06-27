#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class ImprovedFigureEightController:
    def __init__(self):
        rospy.init_node('improved_figure_eight_controller', anonymous=True)
        
        # 参数设置,线速度linear_speed，圆半径circle_radius，角速度angular_speed
        ###########需补全##############
        
        self.circle_radius=0.2
        self.linear_speed=0.6
        self.angular_speed=3.0
        self.period=(math.pi*self.circle_radius/self.linear_speed)*2

        self.start_time=rospy.get_time()
        self.current_angular=self.angular_speed


        # 运动状态
        ###########需补全##############
        self.lock_time=2.0
        self.phase=1
     
        # 机器人当前位置
        self.start_yaw  = None
        self.threshold = 0.05 # 角度误差阈值
        
        self.last_switch_time = rospy.Time.now()
        self.time_enable = False # 是否运行切换的时间标志

        # 创建速度发布器cmd_vel_pub和里程计订阅odom_sub
        ###########需补全##############
        self.cmd_vel_pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10) 
        self.odom_sub=rospy.Subscriber('/odom',Odometry,self.odom_callback,queue_size=10)
        
        # 创建控制定时器（20Hz控制频率）
        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)

        # 注册关闭时的回调函数
        rospy.on_shutdown(self.shutdown_hook)

        rospy.sleep(1)  # 等待里程计数据稳定
        rospy.loginfo("改进版8字轨迹控制器初始化完成")


    def odom_callback(self, odom_msg):
        """更新机器人位置信息"""

        orientation = odom_msg.pose.pose.orientation

        # 从orientation（四元数）获取当前的朝向角current_yaw，可以使用euler_from_quaternion()进行转换
        # 例如(roll, pitch, yaw) = euler_from_quaternion([x, y, z, w])
        ###########需补全##############

  
        (roll, pitch, self.current_yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        
        rospy.loginfo("当前偏航角: %f", self.current_yaw )

        """更新机器人时间信息"""
        current_time = rospy.Time.now()
        elapsed_time = (current_time - self.last_switch_time).to_sec()
        rospy.loginfo("当前执行时间为 %.1f ",elapsed_time)

        # 时间阈值判断
        ###########需补全##############
        if elapsed_time>self.lock_time:
            self.time_enable=True
        else:
            self.time_enable=False

        # 如果是第一次收到位置，设置为起始位置
        ###########需补全##############
        if self.start_yaw is None:
            self.start_yaw=self.current_yaw
            rospy.loginfo("初始化起始方向角:%.2f rad",self.start_yaw)


        # 规范化角度到[-π, π]范围
        def normalize_angle(angle):
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle < -math.pi:
                angle += 2 * math.pi
            return angle
        
        # 计算当前朝向current_yaw和起始朝向start_yaw的角度差angle_diff（考虑周期性）
        ###########需补全##############
        angle_diff=normalize_angle(self.current_yaw-self.start_yaw)
        # 检查是否在角度和时间阈值范围内
        ###########需补全##############
        if abs(angle_diff)<self.threshold and self.time_enable:
            # 改变角速度状态
            ###########需补全##############
            self.current_angular*=-1
            self.phase=3-self.phase
            self.last_switch_time=current_time

            self.start_yaw=self.current_yaw
            rospy.loginfo("切换运动方向，当前偏航角: %f, 与初始角度差: %f", self.current_yaw, angle_diff)
            # 更新起始角度，为下一次切换做准备
            ###########需补全##############

    def publish_stop_signal(self):
        """发布停止信号"""
        rospy.loginfo("发布停止信号...")
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        
        # 多次发布确保收到
        for i in range(5):
            self.cmd_vel_pub.publish(stop_cmd)
            rospy.sleep(0.1)  # 短暂延迟
        
        rospy.loginfo("停止信号发布完成")

    def shutdown_hook(self):
        """ROS关闭时的回调函数"""
        rospy.loginfo("接收到关闭信号，开始清理...")

        # 发布停止信号
        self.publish_stop_signal()
        
        # 取消定时器
        self.control_timer.shutdown()
        
        rospy.loginfo("控制器已安全停止")

    def control_loop(self, event):
        """主控制循环"""
        try:
            # 创建并发布速度指令
            ###########需补全##############
            twist = Twist()
            twist.linear.x = self.linear_speed  # 恒定线速度
            twist.angular.z = self.current_angular
            self.cmd_vel_pub.publish(twist)

        except Exception as e:
            rospy.logerr("控制循环错误: %s", str(e))

    def run(self):
        """主运行函数"""
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = ImprovedFigureEightController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序结束")
