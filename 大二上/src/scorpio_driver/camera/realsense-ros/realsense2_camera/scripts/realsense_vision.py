#!/usr/bin/env python3
import rospy 
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
bridge = CvBridge()
def image_callback(msg):
    try:
        cv_image=bridge.imgmsg_to_cv2(msg,"bgr8")
        cv2.imshow("RealSense Color Image",cv_image)
        cv2.waitKey(1)
    except Exception as e:
        rospy.logerr(e)
if __name__=='__main__':
   rospy.init_node('realsense_vision_node')
   rospy.Subscriber('/camera/color/image_raw',Image,image_callback)
   rospy.spin()
  
