#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import yaml
import sys
import os
import math
import threading
import signal
from geometry_msgs.msg import Pose, Twist, PoseWithCovarianceStamped

# ==========================================
# Core Config: Keypoint Strategy
# ==========================================
FILE_PATH = 'optimized_path.yaml'

# Forward: Auto-record every 1.0m (Sparse)
DIST_FORWARD_AUTO = 1.0   

# Reverse: Min distance between snapshots (prevents duplicate points)
DIST_REVERSE_KEYPOINT = 0.15 

# Thresholds for "Stopped" detection
STOP_VELOCITY_THRESH = 0.01  # m/s
STOP_DURATION_REQ = 0.5      # Seconds to wait before snapshot

class SmartRecorder:
    def __init__(self):
        rospy.init_node('smart_recorder', disable_signals=True)
        
        self.current_pose = None
        self.current_vel = Twist() 
        self.pose_lock = threading.Lock()
        
        # Subscribers
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self._pose_cb)
        rospy.Subscriber('/scorpio_base/cmd_vel', Twist, self._vel_cb) 
        
        self.recorded_waypoints = []
        self.stop_start_time = None
        
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        rospy.logwarn("\nStop signal detected. Saving file...")
        self.save_to_file()
        sys.exit(0)

    def _pose_cb(self, msg):
        with self.pose_lock:
            self.current_pose = msg.pose.pose

    def _vel_cb(self, msg):
        self.current_vel = msg

    def save_to_file(self):
        if not self.recorded_waypoints:
            rospy.logwarn("No points recorded!")
            return
        try:
            with open(FILE_PATH, 'w') as f:
                yaml.dump(self.recorded_waypoints, f, default_flow_style=False)
            rospy.loginfo(f"Successfully saved {len(self.recorded_waypoints)} keypoints to {FILE_PATH}")
        except Exception as e:
            rospy.logerr(f"Save failed: {e}")

    def run(self):
        rospy.loginfo("=== Smart Snapshot Recorder Started ===")
        rospy.loginfo(f"[Forward] Auto-record (Every {DIST_FORWARD_AUTO}m)")
        rospy.loginfo(f"[Reverse] Manual Snapshot (Stop for {STOP_DURATION_REQ}s to capture)")
        
        # [FIXED] Initialize variable here to prevent UnboundLocalError
        last_recorded_pose = None
        
        rate = rospy.Rate(10) # 10Hz
        
        while not rospy.is_shutdown():
            if self.current_pose is None:
                rospy.logwarn_throttle(2.0, "Waiting for AMCL pose...")
                rospy.sleep(0.5)
                continue
            
            with self.pose_lock:
                curr_p = self.current_pose
                # Check direction (reversing if linear.x is negative)
                is_reversing = self.current_vel.linear.x < -0.01
                current_speed = abs(self.current_vel.linear.x)

            # --- Distance Check ---
            dist_from_last = 0.0
            if last_recorded_pose is not None:
                dx = curr_p.position.x - last_recorded_pose['x']
                dy = curr_p.position.y - last_recorded_pose['y']
                dist_from_last = math.sqrt(dx**2 + dy**2)
            else:
                dist_from_last = 999.0 # Force record first point

            should_save = False
            save_reason = ""

            # === Logic Branches ===
            
            # 1. Forward Mode: Auto Sparse Recording
            if not is_reversing:
                self.stop_start_time = None # Reset stop timer
                if dist_from_last > DIST_FORWARD_AUTO:
                    should_save = True
                    save_reason = "FORWARD_AUTO"

            # 2. Reverse Mode: Stop & Snapshot
            else:
                # Only start timer if robot is essentially stopped
                if current_speed < STOP_VELOCITY_THRESH:
                    if self.stop_start_time is None:
                        self.stop_start_time = rospy.Time.now()
                    
                    # Check how long we have been stopped
                    elapsed = (rospy.Time.now() - self.stop_start_time).to_sec()
                    
                    if elapsed > STOP_DURATION_REQ:
                        # Stopped long enough AND moved far enough from last point
                        if dist_from_last > DIST_REVERSE_KEYPOINT:
                            should_save = True
                            save_reason = "REVERSE_SNAPSHOT"
                            self.stop_start_time = None # Reset timer after save
                else:
                    # Robot is moving, reset timer
                    self.stop_start_time = None

            # === Execute Save ===
            if should_save:
                wp = {
                    'x': round(curr_p.position.x, 3),
                    'y': round(curr_p.position.y, 3),
                    'z': round(curr_p.orientation.z, 4),
                    'w': round(curr_p.orientation.w, 4)
                }
                self.recorded_waypoints.append(wp)
                last_recorded_pose = wp
                rospy.loginfo(f"[{save_reason}] Point #{len(self.recorded_waypoints)} Saved (x={wp['x']:.2f}, y={wp['y']:.2f})")
            
            rate.sleep()

if __name__ == '__main__':
    recorder = SmartRecorder()
    recorder.run()