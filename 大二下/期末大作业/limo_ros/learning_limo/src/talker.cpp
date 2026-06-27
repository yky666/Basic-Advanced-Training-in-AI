#include <sstream>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <geometry_msgs/Twist.h>

int main(int argc, char **argv)
{
    // ROS节点初始化
    ros::init(argc, argv, "talker");

	// 创建节点句柄
	ros::NodeHandle n;

	// 创建一个Publisher，发布名为chatter的topic，消息类型为geometry_msgs::Twist
	ros::Publisher chatter_pub = n.advertise<geometry_msgs::Twist>("cmd_vel", 1000);

	// 设置循环的频率
	ros::Rate loop_rate(10);

   for ( int count = 0; count<10;count++ )
   {
   //设置需要发布的速度大小
    geometry_msgs::Twist twist;
    geometry_msgs::Vector3 linear;
    linear.x=0.1;
    linear.y=0;
    linear.z=0;
    geometry_msgs::Vector3 angular;
    angular.x=0;
    angular.y=0;
    angular.z=0;
   
   //将设置好的速度赋值给twist
    twist.linear=linear;
    twist.angular=angular;

	//将设置好的速度发布出去
    chatter_pub.publish(twist);

    // 循环等待回调函数
    ros::spinOnce();

    // 按照循环频率延时
    loop_rate.sleep();

   }

    return 0;

}

