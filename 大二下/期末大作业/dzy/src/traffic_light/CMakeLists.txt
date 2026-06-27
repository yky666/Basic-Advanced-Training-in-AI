cmake_minimum_required(VERSION 3.0.2)
project(traffic_light)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/yolo_traffic_light_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  decision.py
  yolo_detector.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY
  launch
  weights
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
