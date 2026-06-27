#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import actionlib
from math import sqrt, pow
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from visualization_msgs.msg import Marker
from numpy import array, inf

class RRTExplorationReturnHome:
    def __init__(self):
        # ³õÊ¼»¯ROS½Úµã
        rospy.init_node('rrt_exploration_return_home', anonymous=True)
        
        # ¹Ø±Õ½ÚµãÊ±µÄÇåÀí²Ù×÷
        rospy.on_shutdown(self.shutdown)
        
        # ³õÊ¼»¯¹Ø¼ü±äÁ¿
        self.return_pose = None
        self.exploration_complete = False
        self.return_initiated = False
        self.last_frontier_time = rospy.Time.now()
        self.frontier_timeout = rospy.Duration(15.0)  # 15ÃëÄÚÎÞ±ß½çµãÔòÈÏÎªÌ½Ë÷Íê³É
        
        # ³õÊ¼»¯TF¼àÌýÆ÷
        self.tf_listener = tf.TransformListener()
        
        # µÈ´ýmove_baseÐÐ¶¯·þÎñÆ÷
        rospy.loginfo("µÈ´ýmove_baseÐÐ¶¯·þÎñÆ÷...")
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        # 60ÃëÁ¬½Ó³¬Ê±
        if not self.move_base.wait_for_server(rospy.Duration(60)):
            rospy.logerr("ÎÞ·¨Á¬½Óµ½move_baseÐÐ¶¯·þÎñÆ÷")
            rospy.signal_shutdown("ÎÞ·¨Á¬½Óµ½move_base")
            return
        
        rospy.loginfo("ÒÑÁ¬½Óµ½move_base·þÎñÆ÷")
        
        # ¶©ÔÄ±ß½çµã»°Ìâ£¨À´×Ôrrt_explorationµÄfilter½Úµã£©
        rospy.Subscriber('/filtered_points', Marker, self.filtered_points_callback)
        
        # ¼ÇÂ¼·µº½µã£¨³õÊ¼Î»ÖÃ£©
        self.record_return_point()
        
        rospy.loginfo("RRTÌ½Ë÷·µº½ÏµÍ³³õÊ¼»¯Íê³É£¬¿ªÊ¼Ì½Ë÷...")

    def filtered_points_callback(self, msg):
        """
        »Øµ÷º¯Êý£º´¦ÀíÀ´×Ôrrt_explorationµÄ¹ýÂËºó±ß½çµã
        µ±³¤Ê±¼äÃ»ÓÐÊÕµ½ÐÂµÄ±ß½çµãÊ±£¬ÈÏÎªÌ½Ë÷Íê³É
        """
        # Ã¿´ÎÊÕµ½±ß½çµã¸üÐÂ×îºó½ÓÊÕÊ±¼ä
        self.last_frontier_time = rospy.Time.now()
        
        # ¼ì²é±ß½çµãÊýÁ¿£¬Èç¹ûºÜÉÙÒ²ÈÏÎª½Ó½üÌ½Ë÷Íê³É
        if len(msg.points) <= 2 and not self.exploration_complete:
            rospy.loginfo("¼ì²âµ½ÉÙÁ¿±ß½çµã£¬Ì½Ë÷½Ó½üÍê³É")

    def record_return_point(self):
        """¼ÇÂ¼·µº½µã£¨Ì½Ë÷Æðµã£©"""
        rospy.loginfo("¼ÇÂ¼·µº½µãÖÐ...")
        rospy.sleep(2)  # µÈ´ýTFÊ÷ÎÈ¶¨
        
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # »ñÈ¡´Ómapµ½base_linkµÄ±ä»»
                (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
                
                # ´´½¨·µº½µãÎ»×Ë
                self.return_pose = PoseStamped()
                self.return_pose.header.frame_id = "map"
                self.return_pose.pose.position.x = trans[0]
                self.return_pose.pose.position.y = trans[1]
                self.return_pose.pose.position.z = trans[2]
                self.return_pose.pose.orientation.x = rot[0]
                self.return_pose.pose.orientation.y = rot[1]
                self.return_pose.pose.orientation.z = rot[2]
                self.return_pose.pose.orientation.w = rot[3]
                
                rospy.loginfo("·µº½µã¼ÇÂ¼³É¹¦: x=%.2f, y=%.2f", trans[0], trans[1])
                return
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn("³¢ÊÔ %d/%d »ñÈ¡TF±ä»»Ê§°Ü: %s", attempt+1, max_attempts, str(e))
                rospy.sleep(1)
        
        rospy.logerr("ÎÞ·¨»ñÈ¡»úÆ÷ÈËÎ»ÖÃ£¬·µº½µã¼ÇÂ¼Ê§°Ü")
        rospy.loginfo("ÇëÈ·±£µØÍ¼ÒÑ³õÊ¼»¯£¬ÇÒ/mapµ½/base_linkµÄTF±ä»»¿ÉÓÃ")

    def check_exploration_completion(self):
        """
        ¼ì²éÌ½Ë÷ÊÇ·ñÍê³É
        »ùÓÚ±ß½çµã³¬Ê±»úÖÆÅÐ¶Ï
        """
        if self.exploration_complete:
            return True
            
        current_time = rospy.Time.now()
        time_since_last_frontier = current_time - self.last_frontier_time
        
        # Èç¹û³¬¹ý³¬Ê±Ê±¼äÃ»ÓÐÊÕµ½±ß½çµã£¬ÈÏÎªÌ½Ë÷Íê³É
        if time_since_last_frontier > self.frontier_timeout:
            self.exploration_complete = True
            rospy.loginfo("Ì½Ë÷Íê³É£¡%.1fÃëÄÚÎ´¼ì²âµ½ÐÂµÄ±ß½çµã", self.frontier_timeout.to_sec())
            return True
            
        return False

    def navigate_to_point(self, pose):
        """µ¼º½µ½Ö¸¶¨µã"""
        # ´´½¨µ¼º½Ä¿±ê
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        
        # ÉèÖÃÄ¿±êÎ»ÖÃ
        goal.target_pose.pose.position.x = pose.pose.position.x
        goal.target_pose.pose.position.y = pose.pose.position.y
        goal.target_pose.pose.position.z = pose.pose.position.z
        goal.target_pose.pose.orientation = pose.pose.orientation
        
        # ·¢ËÍÄ¿±ê
        self.move_base.send_goal(goal)
        rospy.loginfo("ÒÑ·¢ËÍ·µº½Ä¿±êµã: x=%.2f, y=%.2f", 
                     pose.pose.position.x, pose.pose.position.y)
        
        # µÈ´ýÖ´ÐÐ½á¹û£¬³¬Ê±ÉèÖÃÎª180Ãë
        finished_within_time = self.move_base.wait_for_result(rospy.Duration(180))
        
        if not finished_within_time:
            self.move_base.cancel_goal()
            rospy.logwarn("µ¼º½³¬Ê±£¬È¡Ïûµ±Ç°Ä¿±ê")
            return False
        else:
            state = self.move_base.get_state()
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("³É¹¦·µ»ØÆðÊ¼µã!")
                return True
            else:
                rospy.logwarn("µ¼º½Ê§°Ü£¬×´Ì¬´úÂë: %d", state)
                return False

    def return_to_start(self):
        """Ö´ÐÐ·µº½²Ù×÷"""
        if self.return_pose is None:
            rospy.logerr("Ã»ÓÐÉèÖÃ·µº½µã£¬ÎÞ·¨·µº½")
            return False
            
        rospy.loginfo("¿ªÊ¼·µº½...")
        success = self.navigate_to_point(self.return_pose)
        
        if success:
            rospy.loginfo("·µº½³É¹¦£¡ÈÎÎñÍê³É£¡")
        else:
            rospy.logwarn("·µº½Ê§°Ü£¬½«ÖØÊÔ...")
            
        return success

    def shutdown(self):
        """½Úµã¹Ø±ÕÊ±µÄÇåÀí²Ù×÷"""
        rospy.loginfo("Í£Ö¹»úÆ÷ÈË²¢¹Ø±Õ½Úµã...")
        if hasattr(self, 'move_base'):
            self.move_base.cancel_goal()
        rospy.sleep(1)

    def run(self):
        """Ö÷Ñ­»·"""
        rate = rospy.Rate(1)  # 1Hz¼ì²éÆµÂÊ
        
        while not rospy.is_shutdown():
            # ¼ì²éÌ½Ë÷ÊÇ·ñÍê³É
            if not self.exploration_complete:
                self.check_exploration_completion()
            
            # Èç¹ûÌ½Ë÷Íê³ÉÇÒÉÐÎ´¿ªÊ¼·µº½
            elif self.exploration_complete and not self.return_initiated:
                rospy.loginfo("Ì½Ë÷Íê³É£¬¿ªÊ¼·µº½Á÷³Ì...")
                self.return_initiated = True
                
                # Ö´ÐÐ·µº½
                success = self.return_to_start()
                if success:
                    rospy.loginfo("×ÔÖ÷Ì½Ë÷Óë·µº½ÈÎÎñÔ²ÂúÍê³É£¡")
                    # ÈÎÎñÍê³É£¬¿ÉÒÔ¹Ø±Õ½Úµã»òÖ´ÐÐÆäËû²Ù×÷
                    rospy.signal_shutdown("ÈÎÎñÍê³É")
                else:
                    # ·µº½Ê§°Ü£¬¿ÉÒÔÌí¼ÓÖØÊÔÂß¼­
                    rospy.logwarn("Ê×´Î·µº½³¢ÊÔÊ§°Ü£¬10ÃëºóÖØÊÔ...")
                    rospy.sleep(60)#间隔时间阈值判断
                    self.return_to_start()  # ÖØÊÔÒ»´Î
            
            rate.sleep()

if __name__ == '__main__':
    try:
        return_home = RRTExplorationReturnHome()
        return_home.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("RRTÌ½Ë÷·µº½½ÚµãÒÑ¹Ø±Õ")
