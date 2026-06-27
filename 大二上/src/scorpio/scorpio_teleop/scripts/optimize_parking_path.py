#!/usr/bin/env python3
import yaml
import math
import copy

INPUT_FILE = 'recorded_path.yaml'
OUTPUT_FILE = 'optimized_path.yaml'

def get_yaw_from_quat(z, w):
    return math.atan2(2.0 * (w * z), 1.0 - 2.0 * (z * z))

def get_quat_from_yaw(yaw):
    return {'z': math.sin(yaw / 2.0), 'w': math.cos(yaw / 2.0)}

def optimize_path():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        waypoints = yaml.safe_load(f)

    if not waypoints:
        print("Error: Empty file.")
        return

    # 1. Identify the Final Parking Pose
    final_pt = waypoints[-1]
    final_yaw = get_yaw_from_quat(final_pt['z'], final_pt['w'])
    
    print(f"Final Parking Pose: x={final_pt['x']:.3f}, y={final_pt['y']:.3f}, yaw={math.degrees(final_yaw):.1f} deg")

    # 2. Strict Alignment for the Last Segment (Reverse Entry)
    # We assume the last 15-20 points are the parking maneuver.
    # We will force the last 10 points to have the EXACT SAME orientation as the final point.
    # This prevents the car from entering at an angle.
    ALIGNMENT_COUNT = 15 
    
    # 3. Smooth the path leading into the parking spot
    # We create a "virtual rail" for the last segment
    
    print(f"Optimizing last {ALIGNMENT_COUNT} points for parallel parking...")
    
    # Get the vector of the parking spot based on final yaw
    park_dir_x = math.cos(final_yaw)
    park_dir_y = math.sin(final_yaw)

    for i in range(len(waypoints) - ALIGNMENT_COUNT, len(waypoints)):
        if i < 0: continue
        
        # A. Force Orientation Alignment
        # Overwrite orientation to match the final perfect parking angle
        # This forces the robot to align its body BEFORE it reaches the end
        quat = get_quat_from_yaw(final_yaw)
        waypoints[i]['z'] = round(quat['z'], 4)
        waypoints[i]['w'] = round(quat['w'], 4)
        
        # B. (Optional) Force Position Linearity
        # If you want it perfectly straight, we can project points onto the line defined by the final point
        # Uncomment below to enable strict straight-line enforcing for the last meter
        """
        curr = waypoints[i]
        dx = curr['x'] - final_pt['x']
        dy = curr['y'] - final_pt['y']
        
        # Project vector (dx, dy) onto the parking direction vector
        projection = dx * park_dir_x + dy * park_dir_y
        
        # New position strictly on the parking line
        waypoints[i]['x'] = round(final_pt['x'] + projection * park_dir_x, 3)
        waypoints[i]['y'] = round(final_pt['y'] + projection * park_dir_y, 3)
        """

    # 4. Save Optimized Path
    with open(OUTPUT_FILE, 'w') as f:
        yaml.dump(waypoints, f, default_flow_style=False)
    
    print(f"Done! Saved to {OUTPUT_FILE}.")
    print(f"Please update your navigation script to use '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    optimize_path()