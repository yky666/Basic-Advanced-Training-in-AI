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
import dynamic_reconfigure.client # [NEW] ���붯̬���ÿͻ���
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalStatus

# ==========================================
# Core Configuration
# ==========================================
FILE_PATH = 'optimized_path.yaml'

# [NEW] Dynamic Speed Configuration
# ------------------------------------------------------
# Index threshold: When to switch to "Safe Mode" (Return trip)
# e.g., if path has 60 points, and we set this to 10, 
# it will slow down for the LAST 10 points.
SLOW_DOWN_COUNT = 5 

# FAST PROFILE (Normal Lap)
PARAMS_FAST = {
    'max_vel_x': 1.3,
    'acc_lim_x': 1.8,
    'max_vel_theta': 4.5,
    'acc_lim_theta': 3.6
}

# SAFE PROFILE (Return to Start / Anti-Drift)
PARAMS_SAFE = {
    'max_vel_x': 1.0,   # Slow down significantly
    'acc_lim_x': 1.5,   # Gentle acceleration to prevent slip
    'max_vel_theta': 3.0,
    'acc_lim_theta': 3.0
}
# ------------------------------------------------------

DIST_FORWARD = 0.5        
DIST_REVERSE = 0.15       
TOLERANCE_LOOSE = 0.40    
TOLERANCE_TIGHT = 0.10    
LOOP_CLOSE_THRESHOLD = 3.0
STUCK_DIS_THRESH = 0.05   
STUCK_TIME_THRESH = 2.0   
BACKUP_DISTANCE = 0.20    
BACKUP_SPEED = 0.25       

class PathManager:
    def __init__(self):
        rospy.init_node('path_record_and_nav', disable_signals=True)
        
        self.current_pose = None
        self.current_cmd_x = 0.0
        self.pose_lock = threading.Lock()
        
        # [NEW] Dynamic Reconfigure Client
        # Point this to your move_base TEB planner name
        # Usually: /move_base/TebLocalPlannerROS
        self.teb_client = None
        try:
            self.teb_client = dynamic_reconfigure.client.Client("/move_base/TebLocalPlannerROS", timeout=2)
            rospy.loginfo("Connected to TEB Dynamic Reconfigure Server.")
        except Exception as e:
            rospy.logwarn(f"Could not connect to Dynamic Reconfigure: {e}")
            rospy.logwarn("Speed switching will NOT work.")

        self.current_speed_mode = None # 'fast' or 'safe'

        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self._amcl_pose_callback)
        rospy.Subscriber('/scorpio_base/cmd_vel', Twist, self._cmd_vel_callback)
        self.cmd_vel_pub = rospy.Publisher('/scorpio_base/cmd_vel', Twist, queue_size=10)
        self.initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)

        self.move_base_client = None
        self.stuck_start_time = None
        self.stuck_init_x = 0.0
        self.stuck_init_y = 0.0
        self.need_backup_flag = False
        self.is_goal_active = False
        self.goal_lock = threading.Lock()
        
        self.recorded_waypoints = []
        self.app_mode = 'idle' 
        self.BACKUP_DURATION = BACKUP_DISTANCE / BACKUP_SPEED

        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        rospy.logwarn("\n!!! STOPPING !!!")
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
            rospy.loginfo(f"Saved {len(waypoints)} points.")
        except Exception as e:
            rospy.logerr(f"Save failed: {e}")

    # [NEW] Speed Switching Function
    def set_speed_mode(self, mode):
        if not self.teb_client: return
        if self.current_speed_mode == mode: return # No need to update if same

        rospy.loginfo(f"--- Switching to {mode.upper()} SPEED Mode ---")
        try:
            if mode == 'fast':
                self.teb_client.update_configuration(PARAMS_FAST)
            elif mode == 'safe':
                self.teb_client.update_configuration(PARAMS_SAFE)
            self.current_speed_mode = mode
        except Exception as e:
            rospy.logerr(f"Failed to update parameters: {e}")

    def force_reset_pose(self, start_waypoint):
        rospy.logwarn(">>> [Anti-Drift] Resetting AMCL Pose <<<")
        self.move_base_client.cancel_all_goals()
        for _ in range(5):
            self.cmd_vel_pub.publish(Twist())
            rospy.sleep(0.05)
            
        p = PoseWithCovarianceStamped()
        p.header.frame_id = "map"
        p.header.stamp = rospy.Time.now()
        p.pose.pose.position.x = start_waypoint['x']
        p.pose.pose.position.y = start_waypoint['y']
        p.pose.pose.position.z = 0.0
        p.pose.pose.orientation.x = 0.0
        p.pose.pose.orientation.y = 0.0
        p.pose.pose.orientation.z = start_waypoint['z']
        p.pose.pose.orientation.w = start_waypoint['w']
        p.pose.covariance[0] = 0.05 
        p.pose.covariance[7] = 0.05 
        p.pose.covariance[35] = 0.05
        
        self.initial_pose_pub.publish(p)
        rospy.sleep(0.5)

    def run_record_mode(self):
        # (Keep your existing record code here)
        self.app_mode = 'record'
        rospy.loginfo("=== RECORD MODE ===")
        last_recorded_pose = None
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.current_pose is None:
                rospy.sleep(0.5)
                continue
            with self.pose_lock:
                curr_p = self.current_pose
                cmd_x = self.current_cmd_x
            if cmd_x < -0.01: record_thresh = DIST_REVERSE
            else: record_thresh = DIST_FORWARD
            should_save = False
            if last_recorded_pose is None: should_save = True
            else:
                dx = curr_p.position.x - last_recorded_pose['x']
                dy = curr_p.position.y - last_recorded_pose['y']
                dist = math.sqrt(dx**2 + dy**2)
                dqz = abs(curr_p.orientation.z - last_recorded_pose['z'])
                if dist >= record_thresh or dqz >= 0.2: should_save = True
            if should_save:
                wp = {'x': round(curr_p.position.x, 3), 'y': round(curr_p.position.y, 3), 'z': round(curr_p.orientation.z, 4), 'w': round(curr_p.orientation.w, 4)}
                self.recorded_waypoints.append(wp)
                last_recorded_pose = wp
                rospy.loginfo(f"Point {len(self.recorded_waypoints)} Saved.")
            rate.sleep()

    def run_nav_mode(self):
        self.app_mode = 'nav'
        if not os.path.exists(FILE_PATH):
            rospy.logerr("File not found.")
            return

        with open(FILE_PATH, 'r') as f:
            goals_list = yaml.safe_load(f)
        
        total_points = len(goals_list)
        
        # Auto-Close Loop
        if total_points > 2:
            start_pt = goals_list[0]
            end_pt = goals_list[-1]
            gap = math.sqrt((start_pt['x']-end_pt['x'])**2 + (start_pt['y']-end_pt['y'])**2)
            if gap < LOOP_CLOSE_THRESHOLD:
                goals_list.append(start_pt)
                total_points += 1
                rospy.loginfo("Loop auto-closed.")

        rospy.loginfo(f"=== NAV MODE: {total_points} points ===")
        self.move_base_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        rospy.Timer(rospy.Duration(0.2), self._check_stuck)
        
        lap_count = 0
        
        # Start with FAST mode
        self.set_speed_mode('fast')

        while not rospy.is_shutdown():
            lap_count += 1
            rospy.loginfo(f"\n>>>>>>>> LAP {lap_count} <<<<<<<<\n")
            
            # Reset Pose and Speed at start of lap
            if lap_count > 1:
                self.force_reset_pose(goals_list[0])
                self.set_speed_mode('fast') # Reset to fast for the new lap

            for idx, wp in enumerate(goals_list):
                if rospy.is_shutdown(): return
                
                is_final_goal = (idx == total_points - 1)
                points_remaining = total_points - idx

                # [NEW] Dynamic Speed Logic
                # If we are in the last N points (Return Trip), switch to SAFE mode
                if points_remaining <= SLOW_DOWN_COUNT:
                    self.set_speed_mode('safe')
                else:
                    self.set_speed_mode('fast')

                # Tolerance Logic
                if not is_final_goal:
                    next_wp = goals_list[idx + 1]
                    dx = next_wp['x'] - wp['x']
                    dy = next_wp['y'] - wp['y']
                    segment_dist = math.sqrt(dx**2 + dy**2)
                    if segment_dist > 0.3: current_tolerance = TOLERANCE_LOOSE
                    else: current_tolerance = TOLERANCE_TIGHT
                else:
                    current_tolerance = 0.05 

                success = self.send_goal_with_retry(wp['x'], wp['y'], wp['z'], wp['w'], is_final_goal, current_tolerance)
                if not success: rospy.logwarn(f"Failed Pt {idx+1}")

            rospy.loginfo(f"Lap {lap_count} Done.")
            rospy.sleep(0.5)

    def _check_stuck(self, event):
        if not self.is_goal_active or self.need_backup_flag:
            self.stuck_start_time = None; return
        if self.current_pose is None: return
        with self.pose_lock: cx, cy = self.current_pose.position.x, self.current_pose.position.y
        if self.stuck_start_time is None:
            self.stuck_start_time = rospy.Time.now(); self.stuck_init_x, self.stuck_init_y = cx, cy; return
        dist = math.sqrt((cx - self.stuck_init_x)**2 + (cy - self.stuck_init_y)**2)
        if dist > STUCK_DIS_THRESH:
            self.stuck_start_time = rospy.Time.now(); self.stuck_init_x, self.stuck_init_y = cx, cy
        elif (rospy.Time.now() - self.stuck_start_time).to_sec() > STUCK_TIME_THRESH:
            rospy.logwarn("STUCK -> Backup"); self.need_backup_flag = True; self.move_base_client.cancel_goal(); self.stuck_start_time = None

    def execute_backup(self):
        rospy.sleep(0.5)
        twist = Twist(); twist.linear.x = -BACKUP_SPEED
        start = rospy.Time.now()
        while (rospy.Time.now() - start).to_sec() < self.BACKUP_DURATION:
            if rospy.is_shutdown(): return
            self.cmd_vel_pub.publish(twist); rospy.sleep(0.05)
        self.cmd_vel_pub.publish(Twist()); rospy.sleep(0.5); self.need_backup_flag = False

    def send_goal_with_retry(self, x, y, z, w, is_final_goal, tolerance):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose = Pose(Point(x, y, 0.0), Quaternion(0.0, 0.0, z, w))
        max_retries = 3
        for attempt in range(max_retries + 1):
            if rospy.is_shutdown(): return False
            goal.target_pose.header.stamp = rospy.Time.now()
            with self.goal_lock:
                self.is_goal_active = True; self.need_backup_flag = False; self.stuck_start_time = None
            self.move_base_client.send_goal(goal)
            finished = False
            while not finished:
                if rospy.is_shutdown(): self.move_base_client.cancel_goal(); return False
                if self.need_backup_flag: break 
                if not is_final_goal and self.current_pose:
                    with self.pose_lock: cx, cy = self.current_pose.position.x, self.current_pose.position.y
                    if math.sqrt((cx - x)**2 + (cy - y)**2) < tolerance:
                        self.move_base_client.cancel_goal()
                        with self.goal_lock: self.is_goal_active = False
                        return True 
                finished = self.move_base_client.wait_for_result(rospy.Duration(0.1))
            with self.goal_lock: self.is_goal_active = False
            if self.need_backup_flag: self.execute_backup(); continue 
            if self.move_base_client.get_state() == GoalStatus.SUCCEEDED: return True
            else: rospy.logwarn("Retry..."); rospy.sleep(0.2)
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 record_and_nav_dynamic.py [record|nav]")
        sys.exit(1)
    manager = PathManager()
    mode = sys.argv[1]
    if mode == 'record': manager.run_record_mode()
    elif mode == 'nav': manager.run_nav_mode()