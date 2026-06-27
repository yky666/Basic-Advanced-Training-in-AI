#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
import tf2_ros
import math

class OdometryPublisher:
    def __init__(self):
        rospy.init_node('odometry_publisher')
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        # ÐÂÔö£º¾²Ì¬TF¹ã²¥Æ÷£¬ÓÃÓÚ·¢²¼base_footprint¡úbase_linkµÄ±ä»»
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.vx, self.vy, self.vth = 0.0, 0.0, 0.0
        self.last_time = rospy.Time.now()
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)

        # ·¢²¼base_footprint¡úbase_linkµÄ¾²Ì¬±ä»»
        self.publish_static_tf()
    
    def publish_static_tf(self):
        static_t = TransformStamped()
        static_t.header.stamp = rospy.Time.now()
        static_t.header.frame_id = 'base_footprint'
        static_t.child_frame_id = 'base_link'
        # URDFÖÐbase_footprintÓëbase_linkµÄÏà¶ÔÎ»×ËÊÇ(0,0,0)£¬ËùÒÔÕâÀïtranslationºÍrotation±£³ÖÄ¬ÈÏ
        static_t.transform.translation.x = 0.0
        static_t.transform.translation.y = 0.0
        static_t.transform.translation.z = 0.0
        static_t.transform.rotation.w = 1.0  # ÎÞÐý×ª
        self.static_tf_broadcaster.sendTransform(static_t)
    
    def cmd_vel_callback(self, msg):
        self.vx = msg.linear.x
        self.vth = msg.angular.z
    
    def update_odometry(self):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        if dt > 0:
            delta_x = self.vx * math.cos(self.theta) * dt
            delta_y = self.vx * math.sin(self.theta) * dt
            delta_th = self.vth * dt
            self.x += delta_x
            self.y += delta_y
            self.theta += delta_th

            # ·¢²¼odom¡úbase_footprintµÄ±ä»»
            t = TransformStamped()
            t.header.stamp = current_time
            t.header.frame_id = 'odom'
            t.child_frame_id = 'base_footprint'
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            t.transform.rotation.z = math.sin(self.theta / 2.0)
            t.transform.rotation.w = math.cos(self.theta / 2.0)
            self.tf_broadcaster.sendTransform(t)

            # ·¢²¼Odometry»°Ìâ
            odom = Odometry()
            odom.header.stamp = current_time
            odom.header.frame_id = 'odom'
            odom.pose.pose.position.x = self.x
            odom.pose.pose.position.y = self.y
            odom.pose.pose.orientation.z = math.sin(self.theta / 2.0)
            odom.pose.pose.orientation.w = math.cos(self.theta / 2.0)
            odom.twist.twist.linear.x = self.vx
            odom.twist.twist.angular.z = self.vth
            self.odom_pub.publish(odom)
        self.last_time = current_time

if __name__ == '__main__':
    odom_pub = OdometryPublisher()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        odom_pub.update_odometry()
        rate.sleep()
