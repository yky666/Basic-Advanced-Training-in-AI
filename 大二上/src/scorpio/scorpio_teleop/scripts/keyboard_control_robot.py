#!/usr/bin/env python3
# -*- coding: utf-8 -*
 
import  os
import  sys
import  tty, termios
import roslib
import rospy
from geometry_msgs.msg import Twist

#全局变量
cmd=Twist()
pub=rospy.Publisher('cmd_vel',Twist,queue_size=1)

def keyboardLoop():
	rospy.init_node('teleop')
	
	#初始化监听键盘按钮时间间隔
	rate = rospy.Rate(rospy.get_param('~hz', 10))
    	
    #速度变量
    #慢速
	walk_vel_ = rospy.get_param('walk_vel', 0.1)
    #快速
	yaw_rate_ = rospy.get_param('yaw_rate', 0.5)
    #walk_vel_前后速度
	max_tv = walk_vel_
    # yaw_rate_旋转速度
	max_rv = yaw_rate_
    
    #参数初始化
	speed=0
    # global can_release,can_grasp
    # can_grasp=True
    # can_release=False
    
	print ("使用[WASD][QEZC]控制机器人")
	print ("按[F]退出" )
    	
    #读取按键循环
	while not rospy.is_shutdown():
       	# linux读取键盘按键
		fd = sys.stdin.fileno()
		turn =0
		old_settings = termios.tcgetattr(fd)
       	#不产生回显效果
		old_settings[3] = old_settings[3] & ~termios.ICANON & ~termios.ECHO
       	
		try :
			tty.setraw( fd )
			ch = sys.stdin.read( 1 )
		finally :
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
       	# ch代表获取的键盘按键
		if ch == 'w':
			max_tv = walk_vel_
			speed = 1
			turn = 0
		elif ch == 's':
			max_tv = walk_vel_
			speed = -1
			turn = 0
		elif ch == 'a':
			max_rv = yaw_rate_
			speed = 0
			turn = 1
		elif ch == 'd':
			max_rv = yaw_rate_
			speed = 0
			turn = -1
		elif ch == 'q':
			max_tv = walk_vel_
			max_rv = yaw_rate_
			speed = 1
			turn = 1
		elif ch == 'e':
			max_tv = walk_vel_
			max_rv = yaw_rate_
			speed = 1
			turn = -1
		elif ch == 'z':
			max_tv = walk_vel_
			max_rv = yaw_rate_
			speed = -1
			turn = -1
		elif ch == 'c':
			max_tv = walk_vel_
			max_rv = yaw_rate_
			speed = -1
			turn = 1
		elif ch == 'f':
			exit()
		else:
			max_tv = walk_vel_
			max_rv = yaw_rate_
			speed = 0
			turn = 0
           		
            #发送消息
		cmd.linear.x = speed * max_tv
		cmd.angular.z = turn * max_rv
		pub.publish(cmd)
		rate.sleep()
        #停止机器人
        #stop_robot()

def stop_robot():
	cmd.linear.x = 0.0
	cmd.angular.z = 0.0
	pub.publish(cmd)
 
if __name__ == '__main__':
	try:
		keyboardLoop()
	except rospy.ROSInterruptException:
		pass
