#pragma once

#include <cstdint>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

namespace lift_robot_camera_adapter
{
class Gemini335CanonicalAdapter final : public rclcpp::Node
{
public:
  explicit Gemini335CanonicalAdapter(const rclcpp::NodeOptions & options);

private:
  using Image = sensor_msgs::msg::Image;
  using CameraInfo = sensor_msgs::msg::CameraInfo;

  bool validate_header(const std_msgs::msg::Header & header, const char * stream);
  bool validate_dimensions(uint32_t width, uint32_t height, const char * stream);
  void handle_image(Image::UniquePtr message, bool is_color);
  void try_publish_pair_locked();
  void handle_info(
    CameraInfo::UniquePtr message, const char * stream,
    const rclcpp::Publisher<CameraInfo>::SharedPtr & publisher, bool is_color);
  void publish_static_transform();
  Image::UniquePtr make_phase2_rgbd_preview(
    const Image & color, const Image & depth, const builtin_interfaces::msg::Time & stamp);
  CameraInfo::UniquePtr make_phase2_camera_info(const CameraInfo & source);

  std::string raw_frame_;
  std::string canonical_frame_;
  int64_t publish_period_ns_;
  int64_t preview_period_ns_;
  uint32_t canonical_width_;
  uint32_t canonical_height_;
  std::mutex pair_mutex_;
  Image::UniquePtr pending_color_;
  Image::UniquePtr pending_depth_;
  int64_t last_pair_stamp_ns_{-1};
  int64_t last_preview_stamp_ns_{-1};
  std::atomic<uint64_t> color_published_{0};
  std::atomic<uint64_t> depth_published_{0};
  uint64_t previous_color_count_{0};
  uint64_t previous_depth_count_{0};
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;

  rclcpp::Publisher<Image>::SharedPtr color_pub_;
  rclcpp::Publisher<Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<CameraInfo>::SharedPtr color_info_pub_;
  rclcpp::Publisher<CameraInfo>::SharedPtr depth_info_pub_;
  rclcpp::Publisher<Image>::SharedPtr preview_rgbd_pub_;
  rclcpp::Publisher<CameraInfo>::SharedPtr preview_info_pub_;
  rclcpp::Subscription<Image>::SharedPtr color_sub_;
  rclcpp::Subscription<Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<CameraInfo>::SharedPtr color_info_sub_;
  rclcpp::Subscription<CameraInfo>::SharedPtr depth_info_sub_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster_;
};
}  // namespace lift_robot_camera_adapter
