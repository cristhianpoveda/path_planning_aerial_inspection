// slam_node — minimal ORB-SLAM3 monocular wrapper.
//
// Subscribes to the compressed camera stream, feeds frames to ORB-SLAM3, and
// publishes the camera pose as geometry_msgs/PoseWithCovarianceStamped.
//
// The pose is UP TO SCALE (monocular). The EKF resolves metric scale from
// telemetry velocity + AGL. ORB-SLAM3 provides NO covariance, so a fixed
// diagonal is attached. Nothing is published while tracking is unhealthy --
// the EKF should predict forward rather than fuse an untrustworthy pose.

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "System.h"

class SlamNode : public rclcpp::Node
{
public:
  SlamNode() : Node("slam_node")
  {
    declare_parameter("vocabulary_path", "/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt");
    declare_parameter("settings_path", "");          // camera_calibration.yaml
    declare_parameter("reference_frame", "odom");
    declare_parameter("use_viewer", false);
    // Fixed diagonal covariance in ORB-SLAM scale
    declare_parameter("pose_cov_translation", 0.05);
    declare_parameter("pose_cov_rotation", 0.05);

    const auto voc = get_parameter("vocabulary_path").as_string();
    const auto cfg = get_parameter("settings_path").as_string();
    reference_frame_ = get_parameter("reference_frame").as_string();
    const bool viewer = get_parameter("use_viewer").as_bool();
    cov_t_ = get_parameter("pose_cov_translation").as_double();
    cov_r_ = get_parameter("pose_cov_rotation").as_double();

    if (cfg.empty()) {
      RCLCPP_FATAL(get_logger(), "settings_path is required (camera_calibration.yaml)");
      throw std::runtime_error("missing settings_path");
    }

    // Blocking: loads the ~145 MB vocabulary, takes several seconds.
    RCLCPP_INFO(get_logger(), "loading ORB-SLAM3 (cfg=%s)", cfg.c_str());
    slam_ = std::make_unique<ORB_SLAM3::System>(
        voc, cfg, ORB_SLAM3::System::MONOCULAR, viewer);

    init_start_ = now();

    pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "vo/pose", 10);
    sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        "/camera/image/compressed", rclcpp::SensorDataQoS(),
        std::bind(&SlamNode::onImage, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "slam_node ready");
  }

  ~SlamNode() override
  {
    if (slam_) slam_->Shutdown();
  }

private:
  static const char * stateName(int s)
  {
    switch (s) {
      case -1: return "SYSTEM_NOT_READY";
      case  0: return "NO_IMAGES_YET";
      case  1: return "NOT_INITIALIZED";
      case  2: return "OK";
      case  3: return "RECENTLY_LOST";
      case  4: return "LOST";
      case  5: return "OK_KLT";
      default: return "UNKNOWN";
    }
  }

  void onImage(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    const cv::Mat buf(1, static_cast<int>(msg->data.size()), CV_8UC1,
                      const_cast<uint8_t *>(msg->data.data()));
    const cv::Mat gray = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "jpeg decode failed");
      return;
    }

    const double t = rclcpp::Time(msg->header.stamp).seconds();

    // Core call. Returns Tcw (world -> camera), up to scale.
    Sophus::SE3f Tcw = slam_->TrackMonocular(gray, t);

    // Condition to publish is to have a decent tracking -> 2 == OK
    const int state = slam_->GetTrackingState();

    // Feature counts: tracked map points (3D) and detected keypoints (2D).
    const auto map_points = slam_->GetTrackedMapPoints();
    const auto keypoints  = slam_->GetTrackedKeyPointsUn();
    int n_tracked = 0;
    for (const auto * mp : map_points) if (mp) ++n_tracked;
    const int n_keypoints = static_cast<int>(keypoints.size());

    // Log every state change immediately; otherwise throttle.
    if (state != last_state_) {
      RCLCPP_INFO(get_logger(), "tracking: %s -> %s  (keypoints=%d, map_points=%d)",
                  stateName(last_state_), stateName(state), n_keypoints, n_tracked);
      last_state_ = state;
      if (state == 2 && !ever_initialised_) {
        ever_initialised_ = true;
        RCLCPP_INFO(get_logger(), "initialised after %.1f s",
                    (now() - init_start_).seconds());
      }
    } else {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                           "tracking: %s  keypoints=%d  map_points=%d",
                           stateName(state), n_keypoints, n_tracked);
    }

    if (state != 2) {
      return;   // don't publish an untrusted pose
    }

    // ORB-SLAM3 gives camera-from-world; ROS wants world-from-camera.
    const Sophus::SE3f Twc = Tcw.inverse();
    const Eigen::Vector3f    p = Twc.translation();
    const Eigen::Quaternionf q = Twc.unit_quaternion();

    geometry_msgs::msg::PoseWithCovarianceStamped out;
    out.header.stamp = msg->header.stamp;      // measurement time, not now()
    out.header.frame_id = reference_frame_;

    out.pose.pose.position.x = p.x();
    out.pose.pose.position.y = p.y();
    out.pose.pose.position.z = p.z();
    out.pose.pose.orientation.x = q.x();
    out.pose.pose.orientation.y = q.y();
    out.pose.pose.orientation.z = q.z();
    out.pose.pose.orientation.w = q.w();

    for (int i = 0; i < 3; ++i) out.pose.covariance[i * 6 + i] = cov_t_;
    for (int i = 3; i < 6; ++i) out.pose.covariance[i * 6 + i] = cov_r_;

    pub_->publish(out);
  }

  std::unique_ptr<ORB_SLAM3::System> slam_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
  std::string reference_frame_;
  double cov_t_{}, cov_r_{};
  int last_state_{-99};
  rclcpp::Time init_start_;
  bool ever_initialised_{false};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SlamNode>());
  rclcpp::shutdown();
  return 0;
}
