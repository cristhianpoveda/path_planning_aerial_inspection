cmake_minimum_required(VERSION 3.8)
project(vslam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

set(ORB_SLAM3_DIR $ENV{ORB_SLAM3_ROOT_DIR})

include_directories(
  ${ORB_SLAM3_DIR}
  ${ORB_SLAM3_DIR}/include
  ${ORB_SLAM3_DIR}/include/CameraModels
  ${ORB_SLAM3_DIR}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(slam_node src/slam_node.cpp)
ament_target_dependencies(slam_node
  rclcpp sensor_msgs geometry_msgs nav_msgs cv_bridge tf2_ros)

target_link_libraries(slam_node
  ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so
  ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib/libg2o.so
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
)

install(TARGETS slam_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
