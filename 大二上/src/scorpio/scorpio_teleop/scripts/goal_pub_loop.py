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


# Recording Thresholds
DIST_FORWARD = 0.5        # Forward: Record every 0.5m (Sparse)
DIST_REVERSE = 0.15       # Reverse: Record every 0.15m (High Precision)

# Navigation Tolerance Settings
TOLERANCE_LOOSE = 0.40    # For long segments (Smooth continuous motion)
TOLERANCE_TIGHT = 0.10    # For short segments (Precision maneuvering)
LOOP_CLOSE_THRESHOLD = 3.0 # Distance to auto-close the loop (Meters)

# Stuck Detection Settings
STUCK_DIS_THRESH = 0.05   # Movement < 5cm considered stuck
STUCK_TIME_THRESH = 2.0   # Time to wait before triggering backup
BACKUP_DISTANCE = 0.20    # Distance to reverse during recovery
BACKUP_SPEED = 0.25       # Speed during backup

class PathManager:
    def __init__(self):
        # Disable default signals to handle Ctrl+C manually
        rospy.init_node('path_record_and_nav', disable_signals=True)
        
        self.current_pose = None
        self.current_cmd_x = 0.0
        self.pose_lock = threading.Lock()
        
        # Subscribers
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self._amcl_pose_callback)
        rospy.Subscriber('/scorpio_base/cmd_vel', Twist, self._cmd_vel_callback)
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/scorpio_base/cmd_vel', Twist, queue_size=10)
        self.initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)

        # State Variables
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

        # Register Ctrl+C handler
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C: Stop robot and save file if recording."""
        rospy.logwarn("\n!!! CTRL+C DETECTED - STOPPING !!!")
        
        # Cancel MoveBase
        if self.move_base_client:
            try: self.move_base_client.cancel_all_goals()
            except: pass
        
        # Force Stop
        stop_twist = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_twist)
            rospy.sleep(0.05)
            
        # Save file if in record mode
        if self.app_mode == 'record' and self.recorded_waypoints:
            self.save_to_file(self.recorded_waypoints)
            
        sys.exit(0)

    def _amcl_pose_callback(self, msg):
        with self.pose_lock:
            self.current_pose = msg.pose.pose

    def _cmd_vel_callback(self, msg):
        """Monitor user teleop commands to detect direction."""
        self.current_cmd_x = msg.linear.x

    def save_to_file(self, waypoints):
        try:
            with open(FILE_PATH, 'w') as f:
                yaml.dump(waypoints, f, default_flow_style=False)
            rospy.loginfo(f"SUCCESS: Saved {len(waypoints)} waypoints to {FILE_PATH}")
        except Exception as e:
            rospy.logerr(f"Failed to save file: {e}")

    def force_reset_pose(self, start_waypoint):
        """
        Force AMCL to reset particles to the exact starting coordinates.
        This eliminates accumulated drift after each lap.
        """
        rospy.logwarn(">>> [Anti-Drift] Resetting AMCL Pose to Start Point <<<")
        
        # Ensure robot is stopped
        self.move_base_client.cancel_all_goals()
        for _ in range(5):
            self.cmd_vel_pub.publish(Twist())
            rospy.sleep(0.05)
            
        # Create PoseWithCovarianceStamped message
        p = PoseWithCovarianceStamped()
        p.header.frame_id = "map"
        p.header.stamp = rospy.Time.now()
        
        # Set Position
        p.pose.pose.position.x = start_waypoint['x']
        p.pose.pose.position.y = start_waypoint['y']
        p.pose.pose.position.z = 0.0
        
        # Set Orientation
        p.pose.pose.orientation.x = 0.0
        p.pose.pose.orientation.y = 0.0
        p.pose.pose.orientation.z = start_waypoint['z']
        p.pose.pose.orientation.w = start_waypoint['w']
        
        # Set Covariance (Small value = High confidence)
        p.pose.covariance[0] = 0.05  # x
        p.pose.covariance[7] = 0.05  # y
        p.pose.covariance[35] = 0.05 # theta
        
        self.initial_pose_pub.publish(p)
        rospy.sleep(0.5) # Wait for AMCL to update

    # ==========================================
    # Mode 1: Direction-Aware Recording
    # ==========================================
    def run_record_mode(self):
        self.app_mode = 'record'
        rospy.loginfo("=== RECORD MODE STARTED ===")
        rospy.loginfo(f"Forward -> Save every {DIST_FORWARD}m")
        rospy.loginfo(f"Reverse -> Save every {DIST_REVERSE}m (Precision Mode)")
        
        last_recorded_pose = None
        rate = rospy.Rate(10) # 10Hz
        
        while not rospy.is_shutdown():
            if self.current_pose is None:
                rospy.logwarn_throttle(2.0, "Waiting for AMCL pose...")
                rospy.sleep(0.5)
                continue
            
            with self.pose_lock:
                curr_p = self.current_pose
                cmd_x = self.current_cmd_x
            
            # --- Direction Logic ---
            # If user presses 'back', use high precision
            if cmd_x < -0.01: 
                record_thresh = DIST_REVERSE
                mode_str = "REVERSE"
            else:
                record_thresh = DIST_FORWARD
                mode_str = "FORWARD"

            should_save = False
            
            if last_recorded_pose is None:
                should_save = True
            else:
                # Calculate distance
                dx = curr_p.position.x - last_recorded_pose['x']
                dy = curr_p.position.y - last_recorded_pose['y']
                dist = math.sqrt(dx**2 + dy**2)
                
                # Calculate angle change
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
                rospy.loginfo(f"[{mode_str}] Point {len(self.recorded_waypoints)} Saved.")
            
            rate.sleep()

    # ==========================================
    # Mode 2: Navigation Loop
    # ==========================================
    def run_nav_mode(self):
        self.app_mode = 'nav'
        
        if not os.path.exists(FILE_PATH):
            rospy.logerr(f"File {FILE_PATH} not found.")
            return

        with open(FILE_PATH, 'r') as f:
            goals_list = yaml.safe_load(f)
        
        # --- Logic: Auto-close the loop ---
        if len(goals_list) > 2:
            start_pt = goals_list[0]
            end_pt = goals_list[-1]
            dx = start_pt['x'] - end_pt['x']
            dy = start_pt['y'] - end_pt['y']
            gap = math.sqrt(dx**2 + dy**2)
            
            rospy.loginfo(f"Gap between Start and End: {gap:.3f} m")
            
            if gap < LOOP_CLOSE_THRESHOLD:
                rospy.loginfo("Gap is small. Closing the loop automatically.")
                goals_list.append(start_pt)
            else:
                rospy.logwarn("Gap is too large. Treating as open path.")

        rospy.loginfo(f"=== NAV MODE: Loaded {len(goals_list)} points ===")
        
        self.move_base_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        
        # Background timer for stuck detection
        rospy.Timer(rospy.Duration(0.2), self._check_stuck)
        
        lap_count = 0
        
        # --- Infinite Loop ---
        while not rospy.is_shutdown():
            lap_count += 1
            rospy.loginfo(f"\n>>>>>>>> STARTING LAP {lap_count} <<<<<<<<\n")
            
            # Reset Pose at start of Lap 2+ to fix drift
            if lap_count > 1:
                self.force_reset_pose(goals_list[0])
            
            for idx, wp in enumerate(goals_list):
                if rospy.is_shutdown(): return
                
                # Check if this is the absolute final point of the list
                is_final_goal = (idx == len(goals_list) - 1)
                
                # Determine Tolerance
                if not is_final_goal:
                    # Look ahead to see segment length
                    next_wp = goals_list[idx + 1]
                    dx = next_wp['x'] - wp['x']
                    dy = next_wp['y'] - wp['y']
                    segment_dist = math.sqrt(dx**2 + dy**2)
                    
                    if segment_dist > 0.3:
                        # Long segment -> Loose tolerance (0.4m) -> Smooth
                        current_tolerance = TOLERANCE_LOOSE
                    else:
                        # Short segment -> Tight tolerance (0.1m) -> Precision
                        current_tolerance = TOLERANCE_TIGHT
                else:
                    # Final point (Parking/Loop Close) -> Strict (0.05m)
                    current_tolerance = 0.05 

                success = self.send_goal_with_retry(wp['x'], wp['y'], wp['z'], wp['w'], is_final_goal, current_tolerance)
                
                if not success:
                    rospy.logwarn(f"Failed Waypoint {idx+1}")

            rospy.loginfo(f"Lap {lap_count} Completed.")
            rospy.sleep(0.5)

    def _check_stuck(self, event):
        """Background thread to detect if robot is stuck."""
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
        
        if dist > STUCK_DIS_THRESH:
            self.stuck_start_time = rospy.Time.now()
            self.stuck_init_x, self.stuck_init_y = cx, cy
        else:
            if (rospy.Time.now() - self.stuck_start_time).to_sec() > STUCK_TIME_THRESH:
                rospy.logwarn("STUCK DETECTED -> Triggering Backup")
                self.need_backup_flag = True
                self.move_base_client.cancel_goal()
                self.stuck_start_time = None

    def execute_backup(self):
        """Perform reverse maneuver."""
        rospy.sleep(0.5) # Wait for move_base to stop
        
        twist = Twist()
        twist.linear.x = -BACKUP_SPEED
        
        start = rospy.Time.now()
        while (rospy.Time.now() - start).to_sec() < self.BACKUP_DURATION:
            if rospy.is_shutdown(): return
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.05)
            
        self.cmd_vel_pub.publish(Twist()) # Stop
        rospy.sleep(0.5)
        self.need_backup_flag = False

    def send_goal_with_retry(self, x, y, z, w, is_final_goal, tolerance):
        """Send goal with adaptive tolerance checking."""
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
            
            # Non-blocking wait loop
            finished = False
            while not finished:
                if rospy.is_shutdown():
                    self.move_base_client.cancel_goal()
                    return False
                
                if self.need_backup_flag:
                    break 
                
                # Check Tolerance
                if not is_final_goal and self.current_pose:
                    with self.pose_lock:
                        cx = self.current_pose.position.x
                        cy = self.current_pose.position.y
                    
                    dist = math.sqrt((cx - x)**2 + (cy - y)**2)
                    
                    if dist < tolerance:
                        self.move_base_client.cancel_goal()
                        with self.goal_lock: self.is_goal_active = False
                        return True 

                finished = self.move_base_client.wait_for_result(rospy.Duration(0.1))
            
            with self.goal_lock:
                self.is_goal_active = False
            
            # Handle Backup
            if self.need_backup_flag:
                self.execute_backup()
                continue 
            
            # Handle Result
            state = self.move_base_client.get_state()
            if state == GoalStatus.SUCCEEDED:
                return True
            else:
                if rospy.is_shutdown(): return False
                rospy.logwarn(f"Goal failed (State {state}). Retrying...")
                rospy.sleep(0.2)
        
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 record_and_nav_final.py [record|nav]")
        sys.exit(1)

    manager = PathManager()
    mode = sys.argv[1]

    if mode == 'record':
        manager.run_record_mode()
    elif mode == 'nav':
        manager.run_nav_mode()
    else:
        print(f"Unknown mode: {mode}")