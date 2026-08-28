// slam_node — minimal ORB-SLAM3 monocular wrapper.
//
// Subscribes to the compressed camera stream, feeds frames to ORB-SLAM3, and
// publishes:
//   vo/pose    geometry_msgs/PoseStamped        camera pose in the VO frame `v`
//   vo/status  drone_interfaces/VoStatus        tracking state, epoch, features
//
// The pose is UP TO SCALE (monocular). The EKF resolves metric scale from
// telemetry velocity + relative altitude.
//
// Three contracts the EKF depends on:
//
//  1. vo/status is published on EVERY frame, including unhealthy ones. VO is
//     the EKF's propagation input, so losing it is a propagation gap that must
//     be announced -- not an absence to be inferred from silence.
//
//  2. vo_epoch increments whenever the pose frame may have been rewritten.
//     THREE independent detectors are needed; none is redundant:
//
//       a) map id change   -- catches Atlas map MERGE and map switch. Observed
//                             in this arena with the tracking state remaining
//                             OK throughout, so (b) and (c) both miss it.
//       b) MapChanged()    -- catches in-map loop closure and global BA, which
//                             leave the map id unchanged.
//       c) leaving OK      -- catches reset / new-map-on-loss, where there are
//                             no tracked map points to read an id from.
//
//     The EKF compares vo_epoch for inequality only and re-anchors on any
//     change. Differencing two poses across such an event yields an increment
//     corresponding to no physical motion, and because VO is the propagation
//     input there is no measurement covariance available to reject it with.
//
//  3. No covariance is published. ORB-SLAM3 provides none, and the EKF derives
//     increment noise from Sigma_base * g (filter_design.md 4.3). A fixed
//     diagonal here would be ignored at best and believed at worst.
//
// The pose frame is `vo_world`, NOT `odom`. ORB-SLAM3's world frame is its
// arbitrary first-keyframe frame and is not gravity-aligned; the EKF estimates
// R_n_v to relate the two (filter_design.md 1, 2).

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <drone_interfaces/msg/vo_status.hpp>

#include "System.h"
#include "MapPoint.h"     // MapPoint::GetMap(), isBad() -- both public
#include "Map.h"          // Map::GetId() -- public

class SlamNode : public rclcpp::Node
{
public:
  // ORB-SLAM3 Tracking::eTrackingState
  static constexpr int ST_SYSTEM_NOT_READY = -1;
  static constexpr int ST_NO_IMAGES_YET    =  0;
  static constexpr int ST_NOT_INITIALIZED  =  1;
  static constexpr int ST_OK               =  2;
  static constexpr int ST_RECENTLY_LOST    =  3;
  static constexpr int ST_LOST             =  4;
  static constexpr int ST_OK_KLT           =  5;

  SlamNode() : Node("slam_node")
  {
    declare_parameter("vocabulary_path", "/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt");
    declare_parameter("settings_path", "");          // camera_calibration.yaml
    declare_parameter("vo_frame", "vo_world");       // NOT odom -- see header
    declare_parameter("use_viewer", false);
    declare_parameter("call_shutdown", false);

    const auto voc = get_parameter("vocabulary_path").as_string();
    const auto cfg = get_parameter("settings_path").as_string();
    vo_frame_ = get_parameter("vo_frame").as_string();
    const bool viewer = get_parameter("use_viewer").as_bool();
    call_shutdown_ = get_parameter("call_shutdown").as_bool();

    if (cfg.empty()) {
      RCLCPP_FATAL(get_logger(), "settings_path is required (camera_calibration.yaml)");
      throw std::runtime_error("missing settings_path");
    }

    // Blocking: loads the ~145 MB vocabulary, takes several seconds.
    RCLCPP_INFO(get_logger(), "loading ORB-SLAM3 (cfg=%s)", cfg.c_str());
    slam_ = std::make_unique<ORB_SLAM3::System>(
        voc, cfg, ORB_SLAM3::System::MONOCULAR, viewer);

    init_start_ = now();

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("vo/pose", 10);

    // RELIABLE with depth: the EKF sorts by stamp internally, so its processing
    // order is deterministic given the same SET of messages. Dropping one is
    // what breaks reproducibility under `ros2 bag play`.
    status_pub_ = create_publisher<drone_interfaces::msg::VoStatus>(
        "vo/status", rclcpp::QoS(10).reliable());

    sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        "camera/image/compressed", rclcpp::SensorDataQoS(),
        std::bind(&SlamNode::onImage, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "slam_node ready");
  }

  ~SlamNode() override
  {
    if (slam_ && call_shutdown_) slam_->Shutdown();
    slam_.release();   // deliberately leak: the process is exiting anyway
  }

private:
  static const char * stateName(int s)
  {
    switch (s) {
      case ST_SYSTEM_NOT_READY: return "SYSTEM_NOT_READY";
      case ST_NO_IMAGES_YET:    return "NO_IMAGES_YET";
      case ST_NOT_INITIALIZED:  return "NOT_INITIALIZED";
      case ST_OK:               return "OK";
      case ST_RECENTLY_LOST:    return "RECENTLY_LOST";
      case ST_LOST:             return "LOST";
      case ST_OK_KLT:           return "OK_KLT";
      default:                  return "UNKNOWN";
    }
  }

  static bool poseValid(int s)
  {
    return s == ST_OK || s == ST_OK_KLT;
  }

  void onImage(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    const cv::Mat buf(1, static_cast<int>(msg->data.size()), CV_8UC1,
                      const_cast<uint8_t *>(msg->data.data()));
    const cv::Mat gray = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "jpeg decode failed");
      return;   // no status published: the EKF sees a frame gap, which is true
    }

    const double t = rclcpp::Time(msg->header.stamp).seconds();

    // Core call. Returns Tcw (world -> camera), up to scale.
    Sophus::SE3f Tcw = slam_->TrackMonocular(gray, t);

    const int state = slam_->GetTrackingState();

    // ---- MapChanged() must be called EXACTLY ONCE per frame ----------------
    // Declared in include/System.h:131. Its implementation (src/System.cc:490)
    // latches a static high-water mark against Atlas::GetLastBigChangeIdx(),
    // which returns the CURRENT map's index (Atlas.cc:173). After a map switch
    // the new map's index can sit below the latch, so this alone is not
    // sufficient -- hence the map-id check below. Call it unconditionally: a
    // big change can land while the tracker is briefly unhealthy.
    const bool map_changed = slam_->MapChanged();

    // Feature counts: tracked map points (3D) and detected keypoints (2D).
    //
    // Note on keypoint counts: during NOT_INITIALIZED, Tracking uses
    // mpIniORBextractor built with 5*nFeatures (Tracking.cc:601), so this
    // number is ~5x larger before initialisation than after. Expected, not a
    // fault.
    const auto map_points = slam_->GetTrackedMapPoints();
    const auto keypoints  = slam_->GetTrackedKeyPointsUn();
    int n_tracked = 0;
    for (const auto * mp : map_points) if (mp) ++n_tracked;
    const int n_keypoints = static_cast<int>(keypoints.size());

    // ---- current Atlas map id, via a tracked map point --------------------
    // System::mpAtlas is private, but MapPoint::GetMap() and Map::GetId() are
    // both public, and tracked map points always belong to the active map.
    // This reaches the map id with no patch to ORB-SLAM3.
    //
    // When tracking is unhealthy there may be no usable map point; hold the
    // previous id in that case -- detector (c) covers it anyway.
    long unsigned int map_id = last_map_id_;
    bool got_map_id = false;
    for (auto * mp : map_points) {
      if (mp && !mp->isBad()) {
        ORB_SLAM3::Map * m = mp->GetMap();
        if (m) { map_id = m->GetId(); got_map_id = true; break; }
      }
    }

    const bool map_id_changed = got_map_id && have_map_id_ && (map_id != last_map_id_);
    const bool left_ok = (last_state_ != -99) && poseValid(last_state_) && !poseValid(state);

    if (map_changed || map_id_changed || left_ok) {
      ++vo_epoch_;
      RCLCPP_WARN(get_logger(),
                  "vo_epoch -> %u  (map_changed=%d map_id_changed=%d left_ok=%d, map_id=%lu)",
                  vo_epoch_, static_cast<int>(map_changed),
                  static_cast<int>(map_id_changed), static_cast<int>(left_ok), map_id);
    }
    if (got_map_id) { last_map_id_ = map_id; have_map_id_ = true; }

    // Log every state change immediately; otherwise throttle.
    if (state != last_state_) {
      RCLCPP_INFO(get_logger(),
                  "tracking: %s -> %s  (keypoints=%d, map_points=%d, map_id=%lu)",
                  stateName(last_state_), stateName(state), n_keypoints, n_tracked, map_id);
      if (poseValid(state) && !ever_initialised_) {
        ever_initialised_ = true;
        RCLCPP_INFO(get_logger(), "initialised after %.1f s",
                    (now() - init_start_).seconds());
      }
    } else {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                           "tracking: %s  keypoints=%d  map_points=%d  map_id=%lu  epoch=%u",
                           stateName(state), n_keypoints, n_tracked, map_id, vo_epoch_);
    }
    last_state_ = state;

    // ---- status: ALWAYS, and BEFORE the early return ----------------------
    // The EKF's failure tree (filter_design.md 6) and g_track (4.3) key on
    // this. Publishing only on OK, or only on failure, leaves the EKF unable
    // to distinguish "VO lost" from "camera stream stopped".
    drone_interfaces::msg::VoStatus st;
    st.header.stamp    = msg->header.stamp;   // same stamp as vo/pose
    st.header.frame_id = vo_frame_;
    st.tracking_state  = stateName(state);
    st.vo_epoch        = vo_epoch_;
    st.n_map_points    = static_cast<uint32_t>(n_tracked);
    st.n_keypoints     = static_cast<uint32_t>(n_keypoints);
    st.pose_valid      = poseValid(state);
    status_pub_->publish(st);

    if (!st.pose_valid) {
      return;   // don't publish an untrusted pose
    }

    // ORB-SLAM3 gives camera-from-world; ROS wants world-from-camera.
    const Sophus::SE3f Twc = Tcw.inverse();
    const Eigen::Vector3f    p = Twc.translation();
    const Eigen::Quaternionf q = Twc.unit_quaternion();

    geometry_msgs::msg::PoseStamped out;
    out.header.stamp    = msg->header.stamp;   // measurement time, not now()
    out.header.frame_id = vo_frame_;

    out.pose.position.x = p.x();
    out.pose.position.y = p.y();
    out.pose.position.z = p.z();
    out.pose.orientation.x = q.x();
    out.pose.orientation.y = q.y();
    out.pose.orientation.z = q.z();
    out.pose.orientation.w = q.w();

    pose_pub_->publish(out);
  }

  std::unique_ptr<ORB_SLAM3::System> slam_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<drone_interfaces::msg::VoStatus>::SharedPtr status_pub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
  std::string vo_frame_;
  bool call_shutdown_{false};

  // Monotonic for the life of the process. NEVER reset: the EKF compares it
  // for inequality only, so a reset could make a real discontinuity read as
  // "no change". A node restart resets it to 0, which correctly differs from
  // whatever the EKF last held -> re-anchor, which is right, because a
  // restarted ORB-SLAM3 has a brand-new map.
  uint32_t vo_epoch_{0};

  long unsigned int last_map_id_{0};
  bool have_map_id_{false};

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
