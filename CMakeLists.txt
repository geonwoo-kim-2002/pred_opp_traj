cmake_minimum_required(VERSION 3.8)
project(pred_opp_traj)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pred_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(f1_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rl_switching_mpc_srv REQUIRED)


include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  /usr/local/include   # libgp 헤더
)
link_directories(/usr/local/lib) # libgp 라이브러리

add_executable(pred_opp_traj_srv
  src/main.cpp
  src/detection.cpp
  src/collect_detections.cpp
  src/gpr_opp_traj.cpp
  src/LocalPath/Track.cpp
  src/LocalPath/CubicSpline1D.cpp
  src/LocalPath/CubicSpline2D.cpp
)
ament_target_dependencies(pred_opp_traj_srv
  rclcpp
  std_msgs
  nav_msgs
  sensor_msgs
  pred_msgs
  visualization_msgs
  vision_msgs
  f1_msgs
  tf2
  tf2_geometry_msgs
  rl_switching_mpc_srv
)
target_link_libraries(pred_opp_traj_srv
  gp
)

install(TARGETS
  pred_opp_traj_srv
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY data/lane/
  DESTINATION share/${PROJECT_NAME}/data/lane
)
install(DIRECTORY data/path/
  DESTINATION share/${PROJECT_NAME}/data/path
)
install(DIRECTORY data/raceline/
  DESTINATION share/${PROJECT_NAME}/data/raceline
)

install(DIRECTORY params/
  DESTINATION share/${PROJECT_NAME}/params
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
