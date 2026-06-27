#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
import yaml
import sys
import os
import math
import threading
import signal
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalStatus

# ==========================================
# Core Configuration
# ==========================================
# FILE_PATH = 'recorded_path.yaml'
FILE_PATH = 'optimized_path.yaml'


# Recording Settings
DIST_FORWARD = 0.5        # Forward: Sparse recording
DIST_REVERSE = 0.15      # Reverse: High precision

# [NEW] Adaptive Tolerance Settings
# If the distance to the next point is large (Forward), use loose tolerance.
# If the distance is small (Reverse), use tight tolerance.
TOLERANCE_LOOSE = 0.40    # For Forward driving (Smoothness)
TOLERANCE_TIGHT = 0.10    # For Reversing (Precision)

class PathManager:
    def __init__(self):
        rospy.init_node('path_record_and_nav', disable_signals=True)
        
        self.current_pose = None
        self.current_cmd_x = 0.0
        self.pose_lock = threading.Lock()
        
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self._amcl_pose_callback)
        rospy.Subscriber('/scorpio_base/cmd_vel', Twist, self._cmd_vel_callback)
        self.cmd_vel_pub = rospy.Publisher('/scorpio_base/cmd_vel', Twist, queue_size=10)

        self.move_base_client = None
        self.stuck_start_time = None
        self.stuck_init_x = 0.0
        self.stuck_init_y = 0.0
        self.need_backup_flag = False
        self.is_goal_active = False
        self.goal_lock = threading.Lock()
        
        self.recorded_waypoints = []
        self.app_mode = 'idle' 

        self.STUCK_DIS_THRESH = 0.05    
        self.STUCK_TIME_THRESH = 2.0    
        
        self.BACKUP_DISTANCE = 0.20     
        self.BACKUP_SPEED = 0.25
        self.BACKUP_DURATION = self.BACKUP_DISTANCE / self.BACKUP_SPEED

        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        rospy.logwarn("\n!!! CTRL+C DETECTED !!!")
        if self.move_base_client:
            try: self.move_base_client.cancel_all_goals()
            except: pass
        
        stop_twist = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_twist)
            rospy.sleep(0.05)
            
        if self.app_mode == 'record' and self.recorded_waypoints:
            self.save_to_file(self.recorded_waypoints)
            
        sys.exit(0)

    def _amcl_pose_callback(self, msg):
        with self.pose_lock:
            self.current_pose = msg.pose.pose

    def _cmd_vel_callback(self, msg):
        self.current_cmd_x = msg.linear.x

    def save_to_file(self, waypoints):
        try:
            with open(FILE_PATH, 'w') as f:
                yaml.dump(waypoints, f, default_flow_style=False)
            rospy.loginfo(f"SUCCESS: Saved {len(waypoints)} waypoints.")
        except Exception as e:
            rospy.logerr(f"Failed to save: {e}")

    # ==========================================
    # Record Mode (Direction Aware)
    # ==========================================
    def run_record_mode(self):
        self.app_mode = 'record'
        rospy.loginfo("=== RECORD MODE ===")
        rospy.loginfo(f"Forward -> Every {DIST_FORWARD}m")
        rospy.loginfo(f"Reverse -> Every {DIST_REVERSE}m")
        
        last_recorded_pose = None
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            if self.current_pose is None:
                rospy.sleep(0.5)
                continue
            
            with self.pose_lock:
                curr_p = self.current_pose
                cmd_x = self.current_cmd_x
            
            # Dynamic Threshold
            if cmd_x < -0.01: 
                record_thresh = DIST_REVERSE
            else:
                record_thresh = DIST_FORWARD

            should_save = False
            if last_recorded_pose is None:
                should_save = True
            else:
                dx = curr_p.position.x - last_recorded_pose['x']
                dy = curr_p.position.y - last_recorded_pose['y']
                dist = math.sqrt(dx**2 + dy**2)
                dqz = abs(curr_p.orientation.z - last_recorded_pose['z'])
                
                if dist >= record_thresh or dqz >= 0.2:
                    should_save = True

            if should_save:
                wp = {
                    'x': round(curr_p.position.x, 3),
                    'y': round(curr_p.position.y, 3),
                    'z': round(curr_p.orientation.z, 4),
                    'w': round(curr_p.orientation.w, 4)
                }
                self.recorded_waypoints.append(wp)
                last_recorded_pose = wp
                rospy.loginfo(f"Point {len(self.recorded_waypoints)} Saved.")
            
            rate.sleep()

    # ==========================================
    # Navigation Mode (Adaptive Tolerance)
    # ==========================================
    def run_nav_mode(self):
        self.app_mode = 'nav'
        if not os.path.exists(FILE_PATH):
            rospy.logerr("File not found.")
            return

        with open(FILE_PATH, 'r') as f:
            goals_list = yaml.safe_load(f)
        
        rospy.loginfo(f"=== NAV MODE: Loaded {len(goals_list)} points ===")
        
        self.move_base_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        rospy.Timer(rospy.Duration(0.2), self._check_stuck)
        
        for idx, wp in enumerate(goals_list):
            # idx is 0-based index here
            display_idx = idx + 1
            rospy.loginfo(f"--- Waypoint {display_idx}/{len(goals_list)} ---")
            
            if rospy.is_shutdown(): break
            
            # 1. Determine if Final Goal
            is_final_goal = (display_idx == len(goals_list))
            
            # 2. [NEW] Calculate Distance to NEXT waypoint to determine Tolerance
            # If there is a next point, calculate distance
            if not is_final_goal:
                next_wp = goals_list[idx + 1]
                dx = next_wp['x'] - wp['x']
                dy = next_wp['y'] - wp['y']
                segment_dist = math.sqrt(dx**2 + dy**2)
                
                # If points are far apart (>0.3m), we are in "Forward Mode" -> Use Loose Tolerance
                if segment_dist > 0.3:
                    current_tolerance = TOLERANCE_LOOSE # 0.40m
                else:
                    current_tolerance = TOLERANCE_TIGHT # 0.10m
            else:
                current_tolerance = 0.05 # Final point irrelevant (must stop)

            success = self.send_goal_with_retry(wp['x'], wp['y'], wp['z'], wp['w'], is_final_goal, current_tolerance)
            
            if not success:
                rospy.logwarn(f"Failed Waypoint {display_idx}")

        rospy.loginfo("Navigation Finished.")

    def _check_stuck(self, event):
        if not self.is_goal_active or self.need_backup_flag:
            self.stuck_start_time = None
            return
        if self.current_pose is None: return

        with self.pose_lock:
            cx, cy = self.current_pose.position.x, self.current_pose.position.y
        
        if self.stuck_start_time is None:
            self.stuck_start_time = rospy.Time.now()
            self.stuck_init_x, self.stuck_init_y = cx, cy
            return
        
        dist = math.sqrt((cx - self.stuck_init_x)**2 + (cy - self.stuck_init_y)**2)
        
        if dist > self.STUCK_DIS_THRESH:
            self.stuck_start_time = rospy.Time.now()
            self.stuck_init_x, self.stuck_init_y = cx, cy
        else:
            if (rospy.Time.now() - self.stuck_start_time).to_sec() > self.STUCK_TIME_THRESH:
                rospy.logwarn("STUCK -> BACKUP")
                self.need_backup_flag = True
                self.move_base_client.cancel_goal()
                self.stuck_start_time = None

    def execute_backup(self):
        rospy.sleep(0.5) 
        twist = Twist()
        twist.linear.x = -self.BACKUP_SPEED
        start = rospy.Time.now()
        while (rospy.Time.now() - start).to_sec() < self.BACKUP_DURATION:
            if rospy.is_shutdown(): return
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.05)
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(0.5)
        self.need_backup_flag = False

    # [NEW] Added 'tolerance' parameter
    def send_goal_with_retry(self, x, y, z, w, is_final_goal, tolerance):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose = Pose(Point(x, y, 0.0), Quaternion(0.0, 0.0, z, w))
        
        max_retries = 3
        for attempt in range(max_retries + 1):
            if rospy.is_shutdown(): return False

            goal.target_pose.header.stamp = rospy.Time.now()
            with self.goal_lock:
                self.is_goal_active = True
                self.need_backup_flag = False
                self.stuck_start_time = None
            
            self.move_base_client.send_goal(goal)
            
            finished = False
            while not finished:
                if rospy.is_shutdown():
                    self.move_base_client.cancel_goal()
                    return False
                
                if self.need_backup_flag:
                    break 
                
                # Dynamic Tolerance Logic
                if not is_final_goal and self.current_pose:
                    with self.pose_lock:
                        cx = self.current_pose.position.x
                        cy = self.current_pose.position.y
                    
                    dist_to_goal = math.sqrt((cx - x)**2 + (cy - y)**2)
                    
                    # Use the calculated tolerance passed from run_nav_mode
                    if dist_to_goal < tolerance:
                        # rospy.loginfo(f"Within tolerance ({tolerance}m). Next.")
                        self.move_base_client.cancel_goal()
                        with self.goal_lock: self.is_goal_active = False
                        return True 

                finished = self.move_base_client.wait_for_result(rospy.Duration(0.1))
            
            with self.goal_lock:
                self.is_goal_active = False
            
            if self.need_backup_flag:
                self.execute_backup()
                continue 
            
            state = self.move_base_client.get_state()
            if state == GoalStatus.SUCCEEDED:
                return True
            else:
                if rospy.is_shutdown(): return False
                rospy.logwarn(f"Retry... (State {state})")
                rospy.sleep(0.2)
        
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 record_and_nav_adaptive.py [record|nav]")
        sys.exit(1)

    manager = PathManager()
    mode = sys.argv[1]

    if mode == 'record':
        manager.run_record_mode()
    elif mode == 'nav':
        manager.run_nav_mode()