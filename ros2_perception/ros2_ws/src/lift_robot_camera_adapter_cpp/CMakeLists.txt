cmake_minimum_required(VERSION 3.8)
project(lift_robot_camera_adapter_cpp)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

add_library(gemini335_canonical_adapter_component SHARED
  src/gemini335_canonical_adapter.cpp)
target_compile_features(gemini335_canonical_adapter_component PUBLIC cxx_std_17)
target_compile_options(gemini335_canonical_adapter_component PRIVATE -O3 -Wall -Wextra -Wpedantic)
target_include_directories(gemini335_canonical_adapter_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(gemini335_canonical_adapter_component
  rclcpp rclcpp_components sensor_msgs geometry_msgs tf2_ros)
rclcpp_components_register_node(gemini335_canonical_adapter_component
  PLUGIN "lift_robot_camera_adapter::Gemini335CanonicalAdapter"
  EXECUTABLE gemini335_canonical_adapter_cpp)

install(TARGETS gemini335_canonical_adapter_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
