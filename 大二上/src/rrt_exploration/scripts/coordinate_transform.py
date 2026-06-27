#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseStamped
from apriltag_ros.msg import AprilTagDetectionArray
# Ìæ»»ÎªÄãµÄ¹¦ÄÜ°üÃû£¨ÐèÓëmsgÎÄ¼þËùÔÚ°üÒ»ÖÂ£©
from robot_vision.msg import AprilTagWorldPose  
import threading
from collections import defaultdict

class PersistentAprilTagTransformer:
    def __init__(self):
        # ³õÊ¼»¯ROS½Úµã
        rospy.init_node('persistent_apriltag_transformer', anonymous=True)
        
        # TF2»º³åÇøºÍ¼àÌýÆ÷£¨»º´æ10ÃëÄÚµÄTF±ä»»£©
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # ÅäÖÃ²ÎÊý£¨¿ÉÍ¨¹ýlaunchÎÄ¼þ»ò²ÎÊý·þÎñÆ÷ÐÞ¸Ä£©
        self.input_topic = rospy.get_param('~input_topic', '/tag_detections')  # AprilTag¼ì²âÊäÈë»°Ìâ
        self.output_topic = rospy.get_param('~output_topic', '/apriltag_world_pose')  # ×Ô¶¨ÒåÏûÏ¢Êä³ö»°Ìâ
        self.target_frame = rospy.get_param('~target_frame', 'base_link')  # Ä¿±êÊÀ½ç×ø±êÏµ£¨Èçbase_link/map£©
        self.timeout = rospy.get_param('~transform_timeout', 0.5)  # TF±ä»»³¬Ê±Ê±¼ä£¨Ãë£©
        
        # ´æ´¢TagµÄ×îÐÂ×´Ì¬£¨Ïß³Ì°²È«£©
        self.tag_lock = threading.Lock()
        self.tag_states = defaultdict(dict)  # key: tag_id, value: {'pose': PoseStamped, 'frame_id': str, 'timestamp': rospy.Time}
        
        # ·¢²¼Æ÷£º·¢²¼×Ô¶¨ÒåÏûÏ¢
        self.world_pose_pub = rospy.Publisher(
            self.output_topic, 
            AprilTagWorldPose,
            queue_size=50
        )
        
        # ¶©ÔÄÆ÷£º¶©ÔÄAprilTag¼ì²â½á¹û
        self.tag_sub = rospy.Subscriber(
            self.input_topic,
            AprilTagDetectionArray,
            self.tag_detection_callback,
            queue_size=10
        )
        
        rospy.loginfo("PersistentAprilTagTransformer½Úµã³õÊ¼»¯Íê³É£º")
        rospy.loginfo(f"  - ÊäÈë»°Ìâ£º{self.input_topic}")
        rospy.loginfo(f"  - Êä³ö»°Ìâ£º{self.output_topic}")
        rospy.loginfo(f"  - Ä¿±ê×ø±êÏµ£º{self.target_frame}")
        
    def transform_detection_to_world(self, detection):
        """½«µ¥Ö¡Tag¼ì²â½á¹û×ª»»µ½Ä¿±êÊÀ½ç×ø±êÏµ"""
        try:
            # »ñÈ¡TagÔÚÏà»ú×ø±êÏµÏÂµÄÎ»×Ë£¨apriltag_rosÊä³öµÄÊÇÏà¶ÔÓÚÏà»úµÄPoseStamped£©
            cam_pose = detection.pose
            cam_frame_id = cam_pose.header.frame_id
            
            # µÈ´ýTF±ä»»£¨Ä¿±ê×ø±êÏµ¡úÏà»ú×ø±êÏµ£©
            self.tf_buffer.can_transform(
                target_frame=self.target_frame,
                source_frame=cam_frame_id,
                time=cam_pose.header.stamp,
                timeout=rospy.Duration(self.timeout)
            )
            
            # Ö´ÐÐ×ø±ê±ä»»£¨Ïà»ú×ø±êÏµ¡úÊÀ½ç×ø±êÏµ£©
            world_pose = self.tf_buffer.transform(
                cam_pose,
                target_frame=self.target_frame,
                timeout=rospy.Duration(self.timeout)
            )
            
            # ¹¹ÔìTag¶ÔÓ¦µÄframe_id£¨¸ñÊ½£ºapriltag_<tag_id>£©
            tag_id = detection.id[0]
            tag_frame_id = f"apriltag_{tag_id}"
            
            return world_pose, tag_frame_id
        
        except tf2_ros.LookupException as e:
            rospy.logwarn(f"TF²éÕÒÊ§°Ü£º{str(e)}")
        except tf2_ros.ConnectivityException as e:
            rospy.logwarn(f"TFÁ¬½ÓÊ§°Ü£º{str(e)}")
        except tf2_ros.ExtrapolationException as e:
            rospy.logwarn(f"TFÍâÍÆÊ§°Ü£º{str(e)}")
        except Exception as e:
            rospy.logerr(f"×ø±ê±ä»»Òì³££º{str(e)}")
        
        return None, None
    
    def tag_detection_callback(self, detection_array_msg):
        """´¦ÀíAprilTag¼ì²â½á¹ûµÄÖ÷»Øµ÷º¯Êý"""
        if not detection_array_msg.detections:
            rospy.logdebug("Î´¼ì²âµ½AprilTag")
            return
        
        current_time = rospy.Time.now()
        new_detections = 0
        updated_detections = 0
        
        for detection in detection_array_msg.detections:
            if not detection.id:
                rospy.logwarn("¼ì²âµ½ÎÞIDµÄTag£¬Ìø¹ý")
                continue
                
            tag_id = detection.id[0]
            pose_world, tag_frame_id = self.transform_detection_to_world(detection)
            
            if pose_world is not None and tag_frame_id is not None:
                with self.tag_lock:
                    # ¼ì²éÊÇ·ñÎªÐÂTag»ò¸üÐÂÒÑÓÐTag
                    if tag_id not in self.tag_states:
                        new_detections += 1
                    else:
                        updated_detections += 1
                    
                    # ¸üÐÂTag×´Ì¬
                    self.tag_states[tag_id] = {
                        'pose': pose_world,
                        'frame_id': tag_frame_id,
                        'timestamp': current_time
                    }
                    
                    # ¹¹Ôì×Ô¶¨ÒåÏûÏ¢
                    custom_msg = AprilTagWorldPose()
                    custom_msg.tag_id = tag_id
                    custom_msg.tag_frame_id = tag_frame_id
                    # ¹¹ÔìPointStamped£¨ÊÀ½ç×ø±êÏµÏÂµÄÎ»ÖÃ£©
                    world_point = PointStamped()
                    world_point.header.stamp = current_time
                    world_point.header.frame_id = self.target_frame  # Í³Ò»ÎªÄ¿±êÊÀ½ç×ø±êÏµ
                    world_point.point = pose_world.pose.position
                    custom_msg.world_pose = world_point
                    
                    # ·¢²¼×Ô¶¨ÒåÏûÏ¢
                    self.world_pose_pub.publish(custom_msg)
                
                # ´òÓ¡ÈÕÖ¾£¨±£Áô3Î»Ð¡Êý£¬ÇåÎúÊä³ö£©
                rospy.loginfo(
                    "Tag %d (frame: %s) ÊÀ½ç×ø±ê: (%.3f, %.3f, %.3f)",
                    tag_id, tag_frame_id,
                    world_point.point.x, world_point.point.y, world_point.point.z
                )
        
        # ´òÓ¡Í³¼ÆÐÅÏ¢
        rospy.logdebug(
            "¼ì²â¸üÐÂ£ºÐÂÔö%d¸öTag£¬¸üÐÂ%d¸öTag£¬µ±Ç°ÀÛ¼Æ%d¸öTag",
            new_detections, updated_detections, len(self.tag_states)
        )
    
    def run(self):
        """½ÚµãÖ÷Ñ­»·"""
        rospy.spin()

if __name__ == '__main__':
    try:
        transformer = PersistentAprilTagTransformer()
        transformer.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("½Úµã±»ÖÐ¶Ï£¬ÍË³ö")
    except Exception as e:
        rospy.logerr(f"½ÚµãÒì³£ÍË³ö£º{str(e)}")
