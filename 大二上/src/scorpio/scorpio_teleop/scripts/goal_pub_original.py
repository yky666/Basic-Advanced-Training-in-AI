#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
import threading
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalStatus

class NavigationManager:
    def __init__(self):
        # 1. ��ʼ���ڵ�
        rospy.init_node('navigation_fixed_backup')
        
        # 2. ��ʼ�� MoveBase �ͻ���
        self.move_base_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base server connection...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base server!")
        
        # 3. �ٶȷ��� (���ȷ�ϻ���������ȷ)
        # ��Ȼ��֮ǰ�Ĵ����õ��� /scorpio_base/cmd_vel�����Ǳ��ֲ���
        # ��������˲������볢�Ը�Ϊ /cmd_vel
        self.cmd_vel_pub = rospy.Publisher('/scorpio_base/cmd_vel', Twist, queue_size=10)
        
        # 4. ��λ���߳���
        self.current_pose = None
        self.pose_lock = threading.Lock()
        
        # 5. ��������ز���
        self.stuck_start_time = None
        self.stuck_init_x = 0.0
        self.stuck_init_y = 0.0
        
        # ��ֵ���ã������ 2���� �ƶ�����С�� 5cm�����ж�Ϊ����
        self.STUCK_DIS_THRESH = 0.05    
        self.STUCK_TIME_THRESH = 2.0    
        
        # --- ���ı�־λ ---
        self.need_backup_flag = False   
        
        # ��������
        self.BACKUP_DISTANCE = 0.20     # �Ӵ󵹳����뵽 20cm ȷ������
        self.BACKUP_SPEED = 0.15        # �ٶ�
        self.BACKUP_DURATION = self.BACKUP_DISTANCE / self.BACKUP_SPEED
        
        self.is_goal_active = False
        self.goal_lock = threading.Lock()
        
        # 6. ���� AMCL
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self._amcl_pose_callback)
        
        # 7. ������ⶨʱ�� (0.2s ���һ�Σ����Ƶ��)
        self.check_timer = rospy.Timer(rospy.Duration(0.2), self._check_stuck)

    def _amcl_pose_callback(self, msg):
        with self.pose_lock:
            self.current_pose = msg.pose.pose

    def _check_stuck(self, event):
        """
        [Timer �߳�] ��������ʹ���ȡ��
        """
        # ���û���ڵ����������Ѿ������˵�����־���Ͳ��ټ��
        if not self.is_goal_active or self.need_backup_flag:
            self.stuck_start_time = None
            return
        
        if self.current_pose is None:
            return
        
        with self.pose_lock:
            curr_x = self.current_pose.position.x
            curr_y = self.current_pose.position.y
        
        # ��ʼ����¼��
        if self.stuck_start_time is None:
            self.stuck_start_time = rospy.Time.now()
            self.stuck_init_x = curr_x
            self.stuck_init_y = curr_y
            return
        
        # �����������λ��
        delta_x = curr_x - self.stuck_init_x
        delta_y = curr_y - self.stuck_init_y
        move_distance = (delta_x**2 + delta_y**2)**0.5

        if move_distance > self.STUCK_DIS_THRESH:
            # ֻҪ���ˣ������ü�ʱ�����������ڻ��ƣ�
            self.stuck_start_time = rospy.Time.now()
            self.stuck_init_x = curr_x
            self.stuck_init_y = curr_y
        else:
            # û�������ʱ��
            stuck_duration = (rospy.Time.now() - self.stuck_start_time).to_sec()
            if stuck_duration >= self.STUCK_TIME_THRESH:
                rospy.logwarn(f"Stuck detected ({stuck_duration:.1f}s)! Triggering Backup...")
                
                # 1. ����
                self.need_backup_flag = True
                
                # 2. ȡ��Ŀ�� (����ж����̵߳� wait_for_result)
                self.move_base_client.cancel_goal()
                
                # 3. ���ü�ʱ������ֹ��������
                self.stuck_start_time = None

    def execute_backup(self):
        """
        [���߳�] ִ�е���
        �ؼ��޸������ӵȴ�ʱ�䣬������ MoveBase �������Ȩ
        """
        rospy.logwarn(">>> STARTING BACKUP SEQUENCE <<<")
        
        # [�ؼ�����1] �ȴ� MoveBase ��ȫͣ��
        # ȡ��Ŀ���MoveBase ���ܻ��ڷ�������ָ���ʱ���Ƿ�����ָ��ᱻ����
        # ���������������һ�����ȷ�� MoveBase ���� /cmd_vel
        rospy.sleep(0.5) 
        
        # [�ؼ�����2] ǿ�Ʒ������� 0 �ٶȣ�ȷ�������˾�ֹ
        stop_twist = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_twist)
            rospy.sleep(0.05)

        # [�ؼ�����3] ִ�е���
        backup_twist = Twist()
        backup_twist.linear.x = -self.BACKUP_SPEED
        
        start_time = rospy.Time.now()
        rate = rospy.Rate(20) # 20Hz ����Ƶ��
        
        rospy.loginfo(f"Reversing for {self.BACKUP_DURATION:.1f} seconds...")
        
        while (rospy.Time.now() - start_time).to_sec() < self.BACKUP_DURATION and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(backup_twist)
            rate.sleep()
        
        # [�ؼ�����4] ����������ͣ��
        self.cmd_vel_pub.publish(stop_twist)
        rospy.sleep(0.5) # ��΢ͣ�٣��û�����ס
        
        rospy.loginfo("Backup completed. Flag reset.")
        self.need_backup_flag = False

    def send_goal_with_retry(self, x, y, z, w):
        """
        ���л��˻��Ƶ�Ŀ�귢�ͺ���
        """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose = Pose(Point(x, y, 0.0), Quaternion(0.0, 0.0, z, w))

        max_retries = 3 
        
        for attempt in range(max_retries + 1):
            rospy.loginfo(f"Navigate to ({x:.2f}, {y:.2f}) - Attempt {attempt+1}/{max_retries+1}")
            
            goal.target_pose.header.stamp = rospy.Time.now()
            
            # ״̬����
            with self.goal_lock:
                self.is_goal_active = True
                self.need_backup_flag = False 
                self.stuck_start_time = None 
            
            # ����Ŀ��
            self.move_base_client.send_goal(goal)
            
            # �����ȴ����
            # ��� Timer ������ cancel_goal����������̷���
            self.move_base_client.wait_for_result()
            
            # �������������۳ɹ����Ǳ�ȡ���������״̬
            with self.goal_lock:
                self.is_goal_active = False
            
            # ��ȡ MoveBase ״̬
            state = self.move_base_client.get_state()
            
            # === ���ȼ��ж��߼� ===
            
            # 1. ���ȼ���Ƿ������ǵĴ���������ǵĿ���
            # (��ʹ MoveBase ���� PREEMPTED �� ABORTED��ֻҪ Flag Ϊ�棬���ǿ���)
            if self.need_backup_flag:
                rospy.logwarn("Goal interrupted by STUCK detector.")
                # �����������̣߳�MoveBase �ѱ� Cancel�������� Sleep ����
                self.execute_backup()
                
                # ������ɺ�continue ������һ�� for ѭ����������ͬһ����
                continue 

            # 2. ����ɹ�
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("Goal Reached Successfully!")
                return True
            
            # 3. ����ʧ����� (����·���滮ʧ�ܣ���û��������)
            else:
                rospy.logwarn(f"Goal failed with MoveBase state: {state}. Retrying...")
                rospy.sleep(1.0)
                
        rospy.logerr(f"Failed to reach goal ({x}, {y}) after max retries.")
        return False

if __name__ == '__main__':
    try:
        nav_manager = NavigationManager()
        # �ȴ�һ�о���
        rospy.sleep(1.0) 

        # ���Ӳ���������б�
        navigation_goals = [
            (0.040, 0.022, 0.005, 1.000),       
            (1.535, -0.718, -0.266, 0.964),  
            (1.994, -2.86, -0.697, 0.717),      
            (1.994, -3.211, -0.697, 0.717),     
            (1.994, -3.763, -0.697, 0.717),     
            (1.994, -5.358, -0.697, 0.717), 
            (1.360, -6.179,  0.944, -0.329),    
            (1.119, -6.074, 1.000, 0.011),      
            (-0.215, -6.074, 1.000, 0.000), 
            (-2.815, -6.074, 1.000, 0.000), 
            (-3.267, -6.274, 1.000, 0.000), 
            (-3.267, -5.878, 0.753, 0.658),     
            (-3.264, -4.540, 0.568, 0.823),     
            (-3.008, -3.676, 0.485, 0.875),     
            (-3.178, -3.750, 0.999, 0.051),     
            (-3.474, -3.171, -0.431, 0.903),    
            (-3.185, -2.495, 0.933, 0.359),     
            (-3.071, -2.816, 0.902, 0.431),     
            (-3.290, -3.242, -0.473, 0.881),    
            (-3.251, -3.060, 0.185, 0.983),     
            (-3.612, -3.537, 0.802, 0.597),     
            (-3.537, -3.059, 0.093, 0.996),     
            (-0.614, 1.723, 0.077, 0.997)       
        ]

        # ѭ��ִ���б�
        for idx, goal_data in enumerate(navigation_goals, 1):
            rospy.loginfo(f"\n=== Sequence {idx}/{len(navigation_goals)} ===")
            x, y, z, w = goal_data
            
            success = nav_manager.send_goal_with_retry(x, y, z, w)
            
            if success:
                rospy.sleep(0.5) 
            else:
                rospy.logerr("Skipping to next goal due to failure.")

        rospy.loginfo("\n=== All navigation goals completed! ===")

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation interrupted by user.")