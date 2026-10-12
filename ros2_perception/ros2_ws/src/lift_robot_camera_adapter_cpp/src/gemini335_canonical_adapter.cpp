#include "lift_robot_camera_adapter_cpp/gemini335_canonical_adapter.hpp"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <utility>
#include <chrono>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace lift_robot_camera_adapter
{
namespace
{
constexpr char kCanonicalBodyFrame[] = "camera_link";
constexpr char kCanonicalOpticalFrame[] = "camera_optical_frame";
constexpr char kAuditedRawFrame[] = "gemini335_color_optical_frame";
}

Gemini335CanonicalAdapter::Gemini335CanonicalAdapter(const rclcpp::NodeOptions & options)
: Node("gemini335_canonical_adapter", options)
{
  raw_frame_ = declare_parameter<std::string>("raw_optical_frame", kAuditedRawFrame);
  canonical_frame_ = declare_parameter<std::string>("canonical_optical_frame", kCanonicalOpticalFrame);
  canonical_width_ = static_cast<uint32_t>(declare_parameter<int>("canonical_width", 1920));
  canonical_height_ = static_cast<uint32_t>(declare_parameter<int>("canonical_height", 1080));
  const auto max_hz = declare_parameter<double>("max_image_publish_hz", 30.0);
  const auto preview_hz = declare_parameter<double>("phase2_preview_publish_hz", 5.0);
  if (raw_frame_ != kAuditedRawFrame || canonical_frame_ != kCanonicalOpticalFrame ||
    canonical_width_ != 1920U || canonical_height_ != 1080U ||
    !std::isfinite(max_hz) || max_hz < 1.0 || max_hz > 30.0 ||
    !std::isfinite(preview_hz) || preview_hz < 1.0 || preview_hz > 10.0)
  {
    throw std::invalid_argument("Gemini canonical invariants or publish rate are invalid");
  }
  publish_period_ns_ = static_cast<int64_t>(1.0e9 / max_hz);
  preview_period_ns_ = static_cast<int64_t>(1.0e9 / preview_hz);

  const auto color_raw = declare_parameter<std::string>("color_image.raw_topic", "/gemini335/color/image_raw");
  const auto color_out = declare_parameter<std::string>("color_image.canonical_topic", "/lift_robot/camera/color/image_raw");
  const auto depth_raw = declare_parameter<std::string>("aligned_depth_image.raw_topic", "/gemini335/depth/image_raw");
  const auto depth_out = declare_parameter<std::string>("aligned_depth_image.canonical_topic", "/lift_robot/camera/aligned_depth/image_raw");
  const auto color_info_raw = declare_parameter<std::string>("color_camera_info.raw_topic", "/gemini335/color/camera_info");
  const auto color_info_out = declare_parameter<std::string>("color_camera_info.canonical_topic", "/lift_robot/camera/color/camera_info");
  const auto depth_info_raw = declare_parameter<std::string>("aligned_depth_camera_info.raw_topic", "/gemini335/depth/camera_info");
  const auto depth_info_out = declare_parameter<std::string>("aligned_depth_camera_info.canonical_topic", "/lift_robot/camera/aligned_depth/camera_info");
  const auto preview_rgbd_out = declare_parameter<std::string>("phase2_preview.rgbd_topic", "/lift_robot/camera/phase2_preview/rgbd");
  const auto preview_info_out = declare_parameter<std::string>("phase2_preview.camera_info_topic", "/lift_robot/camera/phase2_preview/color/camera_info");

  auto qos = rclcpp::SensorDataQoS().keep_last(1);
  auto preview_qos = rclcpp::SensorDataQoS().keep_last(1);
  color_pub_ = create_publisher<Image>(color_out, qos);
  depth_pub_ = create_publisher<Image>(depth_out, qos);
  color_info_pub_ = create_publisher<CameraInfo>(color_info_out, qos);
  depth_info_pub_ = create_publisher<CameraInfo>(depth_info_out, qos);
  preview_rgbd_pub_ = create_publisher<Image>(preview_rgbd_out, preview_qos);
  preview_info_pub_ = create_publisher<CameraInfo>(preview_info_out, preview_qos);
  color_sub_ = create_subscription<Image>(color_raw, qos,
    [this](Image::UniquePtr msg) {handle_image(std::move(msg), true);});
  depth_sub_ = create_subscription<Image>(depth_raw, qos,
    [this](Image::UniquePtr msg) {handle_image(std::move(msg), false);});
  color_info_sub_ = create_subscription<CameraInfo>(color_info_raw, qos,
    [this](CameraInfo::UniquePtr msg) {handle_info(std::move(msg), "color_camera_info", color_info_pub_, true);});
  depth_info_sub_ = create_subscription<CameraInfo>(depth_info_raw, qos,
    [this](CameraInfo::UniquePtr msg) {handle_info(std::move(msg), "aligned_depth_camera_info", depth_info_pub_, false);});

  tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);
  publish_static_transform();
  diagnostics_timer_ = create_wall_timer(std::chrono::seconds(10), [this]() {
    const auto color = color_published_.load(std::memory_order_relaxed);
    const auto depth = depth_published_.load(std::memory_order_relaxed);
    RCLCPP_INFO(get_logger(),
      "adapter_health color_total=%lu depth_total=%lu color_10s=%lu depth_10s=%lu",
      color, depth, color - previous_color_count_, depth - previous_depth_count_);
    previous_color_count_ = color;
    previous_depth_count_ = depth;
  });
  RCLCPP_INFO(get_logger(), "native Gemini adapter active at %.3f Hz; pending cart edge is not published", max_hz);
}

bool Gemini335CanonicalAdapter::validate_header(
  const std_msgs::msg::Header & header, const char * stream)
{
  if (header.frame_id != raw_frame_) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
      "dropping %s: unexpected raw frame '%s'", stream, header.frame_id.c_str());
    return false;
  }
  if (header.stamp.sec == 0 && header.stamp.nanosec == 0U) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
      "dropping %s: zero timestamp", stream);
    return false;
  }
  return true;
}

bool Gemini335CanonicalAdapter::validate_dimensions(
  uint32_t width, uint32_t height, const char * stream)
{
  if (width != canonical_width_ || height != canonical_height_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "dropping %s: dimensions %ux%u != %ux%u", stream, width, height,
      canonical_width_, canonical_height_);
    return false;
  }
  return true;
}

void Gemini335CanonicalAdapter::handle_image(
  Image::UniquePtr message, bool is_color)
{
  const char * stream = is_color ? "color_image" : "aligned_depth_image";
  const std::string expected_encoding = is_color ? "rgb8" : "16UC1";
  if (!validate_header(message->header, stream) ||
    !validate_dimensions(message->width, message->height, stream)) {return;}
  if (message->encoding != expected_encoding) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
      "dropping %s: encoding '%s' != '%s'", stream, message->encoding.c_str(), expected_encoding.c_str());
    return;
  }
  message->header.frame_id = canonical_frame_;
  std::lock_guard<std::mutex> lock(pair_mutex_);
  if (is_color) {
    pending_color_ = std::move(message);
  } else {
    pending_depth_ = std::move(message);
  }
  try_publish_pair_locked();
}

void Gemini335CanonicalAdapter::try_publish_pair_locked()
{
  if (!pending_color_ || !pending_depth_) {return;}
  const auto color_stamp = rclcpp::Time(pending_color_->header.stamp).nanoseconds();
  const auto depth_stamp = rclcpp::Time(pending_depth_->header.stamp).nanoseconds();
  const auto skew_ns = std::llabs(color_stamp - depth_stamp);
  constexpr int64_t kMaxPairSkewNs = 50'000'000;
  if (skew_ns > kMaxPairSkewNs) {
    if (color_stamp < depth_stamp) {pending_color_.reset();} else {pending_depth_.reset();}
    return;
  }
  const auto pair_stamp = std::max(color_stamp, depth_stamp);
  const auto minimum_interval_ns = publish_period_ns_ * 9 / 10;
  if (last_pair_stamp_ns_ >= 0 && pair_stamp - last_pair_stamp_ns_ < minimum_interval_ns) {
    return;
  }
  last_pair_stamp_ns_ = pair_stamp;
  const auto preview_minimum_interval_ns = preview_period_ns_ * 9 / 10;
  if (last_preview_stamp_ns_ < 0 ||
    pair_stamp - last_preview_stamp_ns_ >= preview_minimum_interval_ns)
  {
    last_preview_stamp_ns_ = pair_stamp;
    const auto common_stamp = pending_color_->header.stamp;
    auto preview_rgbd = make_phase2_rgbd_preview(*pending_color_, *pending_depth_, common_stamp);
    preview_rgbd_pub_->publish(std::move(preview_rgbd));
  }
  color_pub_->publish(std::move(pending_color_));
  depth_pub_->publish(std::move(pending_depth_));
  color_published_.fetch_add(1, std::memory_order_relaxed);
  depth_published_.fetch_add(1, std::memory_order_relaxed);
}

void Gemini335CanonicalAdapter::handle_info(
  CameraInfo::UniquePtr message, const char * stream,
  const rclcpp::Publisher<CameraInfo>::SharedPtr & publisher, bool is_color)
{
  if (!validate_header(message->header, stream) ||
    !validate_dimensions(message->width, message->height, stream)) {return;}
  message->header.frame_id = canonical_frame_;
  if (is_color) {
    auto preview_info = make_phase2_camera_info(*message);
    preview_info_pub_->publish(std::move(preview_info));
  }
  publisher->publish(std::move(message));
}

Gemini335CanonicalAdapter::Image::UniquePtr Gemini335CanonicalAdapter::make_phase2_rgbd_preview(
  const Image & color, const Image & depth, const builtin_interfaces::msg::Time & stamp)
{
  auto output = std::make_unique<Image>();
  output->header = color.header;
  output->header.stamp = stamp;
  constexpr uint32_t kPreviewScale = 6U;
  output->height = color.height / kPreviewScale;
  output->width = color.width / kPreviewScale;
  output->encoding = "8UC5";
  output->is_bigendian = false;
  constexpr uint32_t kPackedBytesPerPixel = 5U;
  output->step = output->width * kPackedBytesPerPixel;
  output->data.resize(static_cast<size_t>(output->step) * output->height);
  for (uint32_t y = 0; y < output->height; ++y) {
    const auto * color_row = color.data.data() + static_cast<size_t>(y * kPreviewScale) * color.step;
    const auto * depth_row = depth.data.data() + static_cast<size_t>(y * kPreviewScale) * depth.step;
    auto * output_row = output->data.data() + static_cast<size_t>(y) * output->step;
    for (uint32_t x = 0; x < output->width; ++x) {
      const auto color_offset = static_cast<size_t>(x * kPreviewScale) * 3U;
      const auto depth_offset = static_cast<size_t>(x * kPreviewScale) * 2U;
      const auto output_offset = static_cast<size_t>(x) * kPackedBytesPerPixel;
      output_row[output_offset] = color_row[color_offset];
      output_row[output_offset + 1U] = color_row[color_offset + 1U];
      output_row[output_offset + 2U] = color_row[color_offset + 2U];
      output_row[output_offset + 3U] = depth_row[depth_offset];
      output_row[output_offset + 4U] = depth_row[depth_offset + 1U];
    }
  }
  return output;
}

Gemini335CanonicalAdapter::CameraInfo::UniquePtr
Gemini335CanonicalAdapter::make_phase2_camera_info(const CameraInfo & source)
{
  auto output = std::make_unique<CameraInfo>(source);
  constexpr double kPreviewScale = 1.0 / 6.0;
  output->width /= 6U;
  output->height /= 6U;
  output->k[0] *= kPreviewScale;
  output->k[2] *= kPreviewScale;
  output->k[4] *= kPreviewScale;
  output->k[5] *= kPreviewScale;
  output->p[0] *= kPreviewScale;
  output->p[2] *= kPreviewScale;
  output->p[3] *= kPreviewScale;
  output->p[5] *= kPreviewScale;
  output->p[6] *= kPreviewScale;
  output->header.frame_id = canonical_frame_;
  return output;
}

void Gemini335CanonicalAdapter::publish_static_transform()
{
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = now();
  transform.header.frame_id = kCanonicalBodyFrame;
  transform.child_frame_id = kCanonicalOpticalFrame;
  transform.transform.rotation.x = -0.5;
  transform.transform.rotation.y = 0.5;
  transform.transform.rotation.z = -0.5;
  transform.transform.rotation.w = 0.5;
  tf_broadcaster_->sendTransform(transform);
}
}  // namespace lift_robot_camera_adapter

RCLCPP_COMPONENTS_REGISTER_NODE(lift_robot_camera_adapter::Gemini335CanonicalAdapter)
