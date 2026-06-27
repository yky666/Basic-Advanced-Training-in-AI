#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

def stop_robot(pub):
	"""停止机器人的辅助函数"""
	stop_cmd=Twist()
	stop_cmd.linear.x=0.0
	stop_cmd.angular.z=0.0
	pub.publish(stop_cmd)
	rospy.loginfo("机器人已停止")
	
def forward_movement():
	#初始化ROS节点（匿名模式）
	rospy.init_node('akerman_forward_controller',anonymous=True)
	
	#创建cmd_vel话题发布者（队列大小为10）
	cmd_vel_pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10)
	
	#初始化运动指令（Twist消息类型）
	move_cmd=Twist()
	move_cmd.linear.x=0.2
	
	#设置发布频率（10Hz）
	rate=rospy.Rate(10)
	
	#记录开始时间并执行运行
	start_time=rospy.Time.now()
	target_duration=rospy.Duration(3.0)
	
	rospy.loginfo("开始向前直走（0.2m/s）...")
	
	#循环发布指令直到达到目标时长
	while(rospy.Time.now()-start_time) < target_duration:
		cmd_vel_pub.publish(move_cmd)
		rate.sleep()
		
	#运动结束后停止机器人
	stop_robot(cmd_vel_pub)
	rospy.loginfo("任务完成")
	
if __name__ == '__main__':
	try:
		forward_movement()
	except rospy.ROSInterruptException:
		
		#处理节点意外终止
		rospy.logger("节点被中断，正在紧急停止...")
		rospy.init_node('emergency_stop',anonymou=True)
		stop_robot(rospy.Publisher('/cmd_vel',Twist,queue_size=10))
