// Copyright 2022 Kell Ideas sp. z o.o.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <mutex>
#include <chrono>

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/quaternion.hpp> //for conversions
#include "opencv2/video/tracking.hpp" //for kalman filter

#include "yaml-cpp/yaml.h"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "cv_bridge/cv_bridge.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/transform_broadcaster.h"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "image_transport/camera_common.hpp"

#include "aruco_opencv_msgs/msg/aruco_detection.hpp"
#include "aruco_opencv_msgs/msg/board_pose.hpp"

#include "aruco_opencv/utils.hpp"
#include "aruco_opencv/parameters.hpp"


using rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface;

namespace aruco_opencv
{

class ArucoTracker : public rclcpp_lifecycle::LifecycleNode
{
  protected:
  // Parameters
  std::string cam_base_topic_;
  bool image_is_rectified_;
  std::string output_frame_;
  std::string marker_dict_;
  bool transform_poses_;
  bool publish_tf_;
  double marker_size_;
  int image_sub_qos_reliability_;
  int image_sub_qos_durability_;
  int image_sub_qos_depth_;
  std::string image_transport_;
  std::string board_descriptions_path_;
  bool enable_kalman_filter_;

  // For dummy tag node
  // frame ID of tag (from TF)
  std::vector<std::string> tag_frame_ids_;
  // ID of 'detected' tag
  std::vector<int64_t> tag_ids_;
  std::unordered_map<std::string, int64_t> tag_id_map_;

  std::unordered_map<int64_t, cv::KalmanFilter> tracks_;

  // Kalman filter params
  cv::KalmanFilter KF_;         // instantiate Kalman Filter
  
  int nStates_ = 18;            // the number of states
  int nMeasurements_ = 6;       // the number of measured states
  int nInputs_ = 0;             // the number of action control
  double dt_ = 1/15;            // time between measurements (1/FPS). TODO: Use common param 
                                // for camera and aruco node launch, or get rate using topic statistics or similar

  // ROS
  OnSetParametersCallbackHandle::SharedPtr on_set_parameter_callback_handle_;
  rclcpp_lifecycle::LifecyclePublisher<aruco_opencv_msgs::msg::ArucoDetection>::SharedPtr
    detection_pub_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Time last_msg_stamp_;
  bool cam_info_retrieved_ = false;

  // Aruco
  cv::Mat camera_matrix_;
  cv::Mat distortion_coeffs_;
  cv::Mat marker_obj_points_;
  cv::Ptr<cv::aruco::DetectorParameters> detector_parameters_;
  cv::Ptr<cv::aruco::Dictionary> dictionary_;
  std::vector<std::pair<std::string, cv::Ptr<cv::aruco::Board>>> boards_;

  // Thread safety
  std::mutex cam_info_mutex_;

  // Tf2
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

public:
  explicit ArucoTracker(rclcpp::NodeOptions options)
  : LifecycleNode("aruco_tracker", options),
    camera_matrix_(3, 3, CV_64FC1),
    distortion_coeffs_(4, 1, CV_64FC1, cv::Scalar(0)),
    marker_obj_points_(4, 1, CV_32FC3)
  {
    declare_parameters();
  }

  LifecycleNodeInterface::CallbackReturn on_configure(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Configuring");

    detector_parameters_ = cv::aruco::DetectorParameters::create();

    retrieve_parameters();

    if (ARUCO_DICT_MAP.find(marker_dict_) == ARUCO_DICT_MAP.end()) {
      RCLCPP_ERROR_STREAM(get_logger(), "Unsupported dictionary name: " << marker_dict_);
      return LifecycleNodeInterface::CallbackReturn::FAILURE;
    }

    dictionary_ = cv::aruco::getPredefinedDictionary(ARUCO_DICT_MAP.at(marker_dict_));

    if (!board_descriptions_path_.empty()) {
      load_boards();
    }

    update_marker_obj_points();

    if (publish_tf_) {
      tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    }

    detection_pub_ = create_publisher<aruco_opencv_msgs::msg::ArucoDetection>(
      "aruco_detections", 5);
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>("~/debug", 5);

    return LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  LifecycleNodeInterface::CallbackReturn on_activate(const rclcpp_lifecycle::State & state)
  {
    RCLCPP_INFO(get_logger(), "Activating");

    if (transform_poses_) {
      tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }

    LifecycleNode::on_activate(state);

    detection_pub_->on_activate();
    debug_pub_->on_activate();

    on_set_parameter_callback_handle_ =
      add_on_set_parameters_callback(
      std::bind(
        &ArucoTracker::callback_on_set_parameters,
        this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Waiting for first camera info...");

    cam_info_retrieved_ = false;

    std::string cam_info_topic = image_transport::getCameraInfoTopic(cam_base_topic_);
    cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      cam_info_topic, 1,
      std::bind(&ArucoTracker::callback_camera_info, this, std::placeholders::_1));

    rmw_qos_profile_t image_sub_qos = rmw_qos_profile_default;
    image_sub_qos.reliability =
      static_cast<rmw_qos_reliability_policy_t>(image_sub_qos_reliability_);
    image_sub_qos.durability = static_cast<rmw_qos_durability_policy_t>(image_sub_qos_durability_);
    image_sub_qos.depth = image_sub_qos_depth_;

    auto qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(image_sub_qos), image_sub_qos);

    img_sub_ = create_subscription<sensor_msgs::msg::Image>(
      cam_base_topic_, qos, std::bind(
        &ArucoTracker::callback_image, this, std::placeholders::_1));

    return LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  LifecycleNodeInterface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state)
  {
    RCLCPP_INFO(get_logger(), "Deactivating");

    on_set_parameter_callback_handle_.reset();
    cam_info_sub_.reset();
    img_sub_.reset();
    tf_listener_.reset();
    tf_buffer_.reset();

    detection_pub_->on_deactivate();
    debug_pub_->on_deactivate();

    return LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  LifecycleNodeInterface::CallbackReturn on_cleanup(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Cleaning up");

    tf_broadcaster_.reset();
    dictionary_.reset();
    detector_parameters_.reset();
    detection_pub_.reset();
    debug_pub_.reset();

    return LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  LifecycleNodeInterface::CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state)
  {
    RCLCPP_INFO(get_logger(), "Shutting down");

    on_set_parameter_callback_handle_.reset();
    cam_info_sub_.reset();
    img_sub_.reset();
    tf_listener_.reset();
    tf_buffer_.reset();
    tf_broadcaster_.reset();
    detector_parameters_.reset();
    detection_pub_.reset();
    debug_pub_.reset();

    return LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

protected:
  void declare_parameters()
  {
    //std::vector<std::string> default_frame_ids({"tag_0_default_link", "tag_1_default_link"});
    declare_parameter("tag_frame_ids", std::vector<std::string>({"tag_0_default_link", "tag_1_default_link"}));

    //std::vector<int> default_frame_ids({0, 1});
    declare_parameter("tag_ids", std::vector<int>({0, 1}));

    declare_parameter("enable_kalman_filter", false);

    declare_param(*this, "cam_base_topic", "camera/image_raw");
    declare_param(*this, "image_is_rectified", false, false);
    declare_param(*this, "output_frame", "");
    declare_param(*this, "marker_dict", "4X4_50");
    declare_param(
      *this, "image_sub_qos.reliability",
      static_cast<int>(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT));
    declare_param(
      *this, "image_sub_qos.durability",
      static_cast<int>(RMW_QOS_POLICY_DURABILITY_VOLATILE));
    declare_param(*this, "image_sub_qos.depth", 1);
    declare_param(*this, "publish_tf", true, true);
    declare_param(*this, "marker_size", 0.15, true);
    declare_param(*this, "board_descriptions_path", "");

    declare_aruco_parameters(*this);
  }

  void retrieve_parameters()
  {
    tag_frame_ids_ = get_parameter("tag_frame_ids").as_string_array();
    tag_ids_ = get_parameter("tag_ids").as_integer_array();

    for (int i = 0; i< tag_frame_ids_.size(); i++)
    {
      tag_id_map_[tag_frame_ids_[i]] = tag_ids_[i];
    }

    get_parameter("enable_kalman_filter", enable_kalman_filter_);

    get_param(*this, "cam_base_topic", cam_base_topic_, "Camera Base Topic: ");

    get_parameter("image_is_rectified", image_is_rectified_);
    RCLCPP_INFO_STREAM(
      get_logger(), "Assume images are rectified: " << (image_is_rectified_ ? "YES" : "NO"));

    get_parameter("output_frame", output_frame_);
    if (output_frame_.empty()) {
      RCLCPP_INFO(get_logger(), "Marker detections will be published in the camera frame");
      transform_poses_ = false;
    } else {
      RCLCPP_INFO(
        get_logger(), "Marker detections will be transformed to \'%s\' frame",
        output_frame_.c_str());
      transform_poses_ = true;
    }

    get_param(*this, "marker_dict", marker_dict_, "Marker Dictionary name: ");

    get_parameter("image_sub_qos.reliability", image_sub_qos_reliability_);
    get_parameter("image_sub_qos.durability", image_sub_qos_durability_);
    get_parameter("image_sub_qos.depth", image_sub_qos_depth_);

    get_parameter("publish_tf", publish_tf_);
    RCLCPP_INFO_STREAM(get_logger(), "TF publishing is " << (publish_tf_ ? "enabled" : "disabled"));

    get_param(*this, "marker_size", marker_size_, "Marker size: ");

    get_parameter("board_descriptions_path", board_descriptions_path_);

    RCLCPP_INFO(get_logger(), "Aruco Parameters:");
    retrieve_aruco_parameters(*this, detector_parameters_, true);
  }

  rcl_interfaces::msg::SetParametersResult callback_on_set_parameters(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    // Validate parameters
    for (auto & param : parameters) {
      if (param.get_name() == "marker_size") {
        if (param.as_double() <= 0.0) {
          result.successful = false;
          result.reason = param.get_name() + " must be positive";
          RCLCPP_ERROR_STREAM(get_logger(), result.reason);
          return result;
        }
      }
    }

    bool aruco_param_changed = false;
    for (auto & param : parameters) {
      if (param.get_name() == "marker_size") {
        marker_size_ = param.as_double();
        update_marker_obj_points();
      } else if (param.get_name().rfind("aruco", 0) == 0) {
        aruco_param_changed = true;
      } else {
        // Unknown parameter, ignore
        continue;
      }

      RCLCPP_INFO_STREAM(
        get_logger(),
        "Parameter \"" << param.get_name() << "\" changed to " << param.value_to_string());
    }

    if (!aruco_param_changed) {
      return result;
    }

    retrieve_aruco_parameters(*this, detector_parameters_);

    return result;
  }

  void load_boards()
  {
    RCLCPP_INFO_STREAM(
      get_logger(), "Trying to load board descriptions from " << board_descriptions_path_);

    YAML::Node descriptions;
    try {
      descriptions = YAML::LoadFile(board_descriptions_path_);
    } catch (const YAML::Exception & e) {
      RCLCPP_ERROR_STREAM(get_logger(), "Failed to load board descriptions: " << e.what());
      return;
    }

    if (!descriptions.IsSequence()) {
      RCLCPP_ERROR(get_logger(), "Failed to load board descriptions: root node is not a sequence");
    }

    for (const YAML::Node & desc : descriptions) {
      std::string name;
      try {
        name = desc["name"].as<std::string>();
        const bool frame_at_center = desc["frame_at_center"].as<bool>();
        const int markers_x = desc["markers_x"].as<int>();
        const int markers_y = desc["markers_y"].as<int>();
        const double marker_size = desc["marker_size"].as<double>();
        const double separation = desc["separation"].as<double>();

        auto board = cv::aruco::GridBoard::create(
          markers_x, markers_y, marker_size, separation,
          dictionary_, desc["first_id"].as<int>());

        if (frame_at_center) {
          double offset_x = (markers_x * (marker_size + separation) - separation) / 2.0;
          double offset_y = (markers_y * (marker_size + separation) - separation) / 2.0;
          for (auto & obj : board->objPoints) {
            for (auto & point : obj) {
              point.x -= offset_x;
              point.y -= offset_y;
            }
          }
        }

        boards_.push_back(std::make_pair(name, board));
      } catch (const YAML::Exception & e) {
        RCLCPP_ERROR_STREAM(get_logger(), "Failed to load board '" << name << "': " << e.what());
        continue;
      }
      RCLCPP_ERROR_STREAM(
        get_logger(), "Successfully loaded configuration for board '" << name << "'");
    }
  }

  void update_marker_obj_points()
  {
    // set coordinate system in the middle of the marker, with Z pointing out
    marker_obj_points_.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-marker_size_ / 2.f, marker_size_ / 2.f, 0);
    marker_obj_points_.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(marker_size_ / 2.f, marker_size_ / 2.f, 0);
    marker_obj_points_.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(marker_size_ / 2.f, -marker_size_ / 2.f, 0);
    marker_obj_points_.ptr<cv::Vec3f>(0)[3] =
      cv::Vec3f(-marker_size_ / 2.f, -marker_size_ / 2.f, 0);
  }

  void callback_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr cam_info)
  {
    std::lock_guard<std::mutex> guard(cam_info_mutex_);

    if (image_is_rectified_) {
      for (int i = 0; i < 12; ++i) {
        camera_matrix_.at<double>(i / 4, i % 4) = cam_info->p[i];
      }
    } else {
      for (int i = 0; i < 9; ++i) {
        camera_matrix_.at<double>(i / 3, i % 3) = cam_info->k[i];
      }
      distortion_coeffs_ = cv::Mat(cam_info->d, true);
    }

    if (!cam_info_retrieved_) {
      RCLCPP_INFO(get_logger(), "First camera info retrieved.");
      cam_info_retrieved_ = true;
    }
  }

  void initKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt)
  {
    KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-3));       // set process noise
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-3));   // set measurement noise
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance
                  /* DYNAMIC MODEL */
    //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]
    // position
    KF.transitionMatrix.at<double>(0,3) = dt;
    KF.transitionMatrix.at<double>(1,4) = dt;
    KF.transitionMatrix.at<double>(2,5) = dt;
    KF.transitionMatrix.at<double>(3,6) = dt;
    KF.transitionMatrix.at<double>(4,7) = dt;
    KF.transitionMatrix.at<double>(5,8) = dt;
    KF.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);
    // orientation
    KF.transitionMatrix.at<double>(9,12) = dt;
    KF.transitionMatrix.at<double>(10,13) = dt;
    KF.transitionMatrix.at<double>(11,14) = dt;
    KF.transitionMatrix.at<double>(12,15) = dt;
    KF.transitionMatrix.at<double>(13,16) = dt;
    KF.transitionMatrix.at<double>(14,17) = dt;
    KF.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);
        /* MEASUREMENT MODEL */
    //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
    KF.measurementMatrix.at<double>(0,0) = 1;  // x
    KF.measurementMatrix.at<double>(1,1) = 1;  // y
    KF.measurementMatrix.at<double>(2,2) = 1;  // z
    KF.measurementMatrix.at<double>(3,9) = 1;  // roll
    KF.measurementMatrix.at<double>(4,10) = 1; // pitch
    KF.measurementMatrix.at<double>(5,11) = 1; // yaw
  }

  void fillMeasurements( cv::Mat &measurements,
                    const cv::Vec3d &translation_measured, const cv::Vec3d &rotation_measured)
  {
      // Convert rotation matrix to euler angles
      // cv::Mat measured_eulers(3, 1, CV_64F);
      // measured_eulers = rotation_measured;
      // Set measurement to predict
      measurements.at<double>(0) = translation_measured[0]; // x
      measurements.at<double>(1) = translation_measured[1]; // y
      measurements.at<double>(2) = translation_measured[2]; // z
      measurements.at<double>(3) = rotation_measured[0];      // roll
      measurements.at<double>(4) = rotation_measured[1];      // pitch
      measurements.at<double>(5) = rotation_measured[2];      // yaw
  }

  void updateKalmanFilter( cv::KalmanFilter &KF, cv::Mat &measurement,
                      cv::Vec3d &translation_estimated, cv::Vec3d &rotation_estimated )
  {
      // First predict, to update the internal statePre variable
      cv::Mat prediction = KF.predict();
      // The "correct" phase that is going to use the predicted value and our measurement
      cv::Mat estimated = KF.correct(measurement);
      // Estimated translation
      translation_estimated[0] = estimated.at<double>(0);
      translation_estimated[1] = estimated.at<double>(1);
      translation_estimated[2] = estimated.at<double>(2);
      // Estimated euler angles
      //cv::Mat eulers_estimated(3, 1, CV_64F);
      rotation_estimated[0] = estimated.at<double>(9);
      rotation_estimated[1] = estimated.at<double>(10);
      rotation_estimated[2] = estimated.at<double>(11);
      // Convert estimated quaternion to rotation matrix
      //rotation_estimated = euler2rot(eulers_estimated);
  }


  void callback_image(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
  {
    RCLCPP_DEBUG_STREAM(get_logger(), "Image message address [SUBSCRIBE]:\t" << img_msg.get());

    if (!cam_info_retrieved_) {
      return;
    }

    if (img_msg->header.stamp == last_msg_stamp_) {
      RCLCPP_DEBUG(
        get_logger(),
        "The new image has the same timestamp as the previous one (duplicate frame?). Ignoring...");
      return;
    }
    last_msg_stamp_ = img_msg->header.stamp;

    auto callback_start_time = get_clock()->now();

    // Convert the image
    auto cv_ptr = cv_bridge::toCvShare(img_msg);

    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;

    // TODO(bjsowa): mutex
    cv::aruco::detectMarkers(
      cv_ptr->image, dictionary_, marker_corners, marker_ids,
      detector_parameters_);

    int n_markers = marker_ids.size();
    std::vector<cv::Vec3d> rvec_final(n_markers), tvec_final(n_markers);
    std::vector<cv::Vec3d> rvec_est_final(n_markers), tvec_est_final(n_markers);

    aruco_opencv_msgs::msg::ArucoDetection detection;
    detection.header.frame_id = img_msg->header.frame_id;
    detection.header.stamp = img_msg->header.stamp;
    detection.markers.resize(n_markers);
    cv::Vec3d tvec_est;
    cv::Vec3d rvec_est;
    {
      std::lock_guard<std::mutex> guard(cam_info_mutex_);

      cv::parallel_for_(
        cv::Range(0, n_markers), [&](const cv::Range & range) {
          for (size_t i = range.start; i < range.end; i++) {
            int id = marker_ids[i];

            // calculate marker poses in camera frame
            cv::solvePnP(
              marker_obj_points_, marker_corners[i], camera_matrix_, distortion_coeffs_,
              rvec_final[i], tvec_final[i], false, cv::SOLVEPNP_IPPE_SQUARE);

            detection.markers[i].marker_id = id;
            detection.markers[i].pose = convert_rvec_tvec(rvec_final[i], tvec_final[i]);

            // add markers to tracker
            if (enable_kalman_filter_) {
              // check if tag is being tracked already
              if (tracks_.count(id) == 0){
                // tag is not being tracked, create KF for it
                // tracks_[id] calls the default constructor for the KalmanFilter class and adds it to the map
                // with the id key
                RCLCPP_INFO(
                  get_logger(),
                  "Start tracking new tag with ID %d", id);
                initKalmanFilter(tracks_[id], nStates_, nMeasurements_, nInputs_, dt_);
              }
              // update filter for tracked IDs 
              cv::Mat measurements(nMeasurements_, 1, CV_64FC1); measurements.setTo(cv::Scalar(0));
              fillMeasurements(measurements, tvec_final[i], rvec_final[i]);
              updateKalmanFilter( tracks_[id], measurements,
                        tvec_est_final[i], rvec_est_final[i]);
            }
          }
        });

      for (const auto & board_desc : boards_) {
        std::string name = board_desc.first;
        auto & board = board_desc.second;

        cv::Vec3d rvec, tvec;
        int valid = cv::aruco::estimatePoseBoard(
          marker_corners, marker_ids, board, camera_matrix_,
          distortion_coeffs_, rvec, tvec);

        if (valid > 0) {
          aruco_opencv_msgs::msg::BoardPose bpose;
          bpose.board_name = name;
          bpose.pose = convert_rvec_tvec(rvec, tvec);
          detection.boards.push_back(bpose);
          rvec_final.push_back(rvec);
          tvec_final.push_back(tvec);
          n_markers++;
        }
      }
    }
    
    if (transform_poses_ && n_markers > 0) {
      detection.header.frame_id = output_frame_;
      geometry_msgs::msg::TransformStamped cam_to_output;
      // Retrieve camera -> output_frame transform
      try {
        cam_to_output = tf_buffer_->lookupTransform(
          output_frame_, img_msg->header.frame_id,
          img_msg->header.stamp, rclcpp::Duration::from_seconds(1.0));
      } catch (tf2::TransformException & ex) {
        RCLCPP_ERROR_STREAM(get_logger(), ex.what());
        return;
      }
      for (auto & marker_pose : detection.markers) {
        tf2::doTransform(marker_pose.pose, marker_pose.pose, cam_to_output);
      }
      for (auto & board_pose : detection.boards) {
        tf2::doTransform(board_pose.pose, board_pose.pose, cam_to_output);
      }
    }

    if (publish_tf_ && n_markers > 0) {
      std::vector<geometry_msgs::msg::TransformStamped> transforms;
      for (auto & marker_pose : detection.markers) {
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = detection.header.stamp;
        transform.header.frame_id = detection.header.frame_id;
        transform.child_frame_id = std::string("marker_") + std::to_string(marker_pose.marker_id);
        tf2::Transform tf_transform;
        tf2::fromMsg(marker_pose.pose, tf_transform);
        transform.transform = tf2::toMsg(tf_transform);
        transforms.push_back(transform);
      }
      for (auto & board_pose : detection.boards) {
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = detection.header.stamp;
        transform.header.frame_id = detection.header.frame_id;
        transform.child_frame_id = std::string("board_") + board_pose.board_name;
        tf2::Transform tf_transform;
        tf2::fromMsg(board_pose.pose, tf_transform);
        transform.transform = tf2::toMsg(tf_transform);
        transforms.push_back(transform);
      }
      tf_broadcaster_->sendTransform(transforms);
    }

    detection_pub_->publish(detection);
    if (debug_pub_->get_subscription_count() > 0) {
      auto debug_cv_ptr = cv_bridge::toCvCopy(img_msg, "bgr8");
      cv::aruco::drawDetectedMarkers(debug_cv_ptr->image, marker_corners, marker_ids);
      {
        std::lock_guard<std::mutex> guard(cam_info_mutex_);
        if(!enable_kalman_filter_) {
          for (size_t i = 0; i < n_markers; i++) {
            cv::drawFrameAxes(
              debug_cv_ptr->image, camera_matrix_, distortion_coeffs_, rvec_final[i],
              tvec_final[i], 0.02, 3);
          }
        }
        else {
          for (size_t i = 0; i < n_markers; i++) {
            try {
              cv::drawFrameAxes(
                debug_cv_ptr->image, camera_matrix_, distortion_coeffs_, rvec_est_final[i],
                tvec_est_final[i], 0.02, 3);
            }
            catch (cv::Exception& ex)
            {
              RCLCPP_INFO_STREAM(
              get_logger(), "Could not draw " << n_markers << " tag(s) axes with " << tvec_est_final.size() << " tvec(s) and " << rvec_est_final.size() << " rvecs because: " << ex.what());
            return;
            }
          }
        }
      }
      std::unique_ptr<sensor_msgs::msg::Image> debug_img =
        std::make_unique<sensor_msgs::msg::Image>();
      debug_cv_ptr->toImageMsg(*debug_img);
      debug_pub_->publish(std::move(debug_img));
    }

    auto callback_end_time = get_clock()->now();
    double whole_callback_duration = (callback_end_time - callback_start_time).seconds();
    double image_send_duration = (callback_start_time - img_msg->header.stamp).seconds();

    RCLCPP_DEBUG(
      get_logger(), "Image callback completed. The callback started %.4f s after the image"
      " frame was grabbed and completed its execution in %.4f s.", image_send_duration,
      whole_callback_duration);
  }
};

class ArucoTrackerAutostart : public ArucoTracker
{
public:
  explicit ArucoTrackerAutostart(rclcpp::NodeOptions options)
  : ArucoTracker(options)
  {
    auto new_state = configure();
    if (new_state.label() == "inactive") {
      activate();
    }
  }
};

class ArucoTrackerDummy : public ArucoTracker
{
rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr tag_0_pose_sub_;
rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr tag_1_pose_sub_;

void getTagTransforms(std::string source_frame_id, std::vector<std::string> tag_frame_ids, std::vector<cv::Vec3d>& tVecs, std::vector<cv::Vec3d>& rVecs)
{
  for( std::string tag_frame_id: tag_frame_ids )
  {
    // Look up for the transformation between tag frame and source frame
      geometry_msgs::msg::TransformStamped t;
      try {
        t = tf_buffer_->lookupTransform(
          source_frame_id, tag_frame_id,
          tf2::TimePointZero);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_INFO(
          get_logger(), "Could not transform %s to %s: %s",
          source_frame_id.c_str(), tag_frame_id.c_str(), ex.what());
        return;
      }
      //RCLCPP_INFO(
      //    get_logger(), "Got Transform for %s: x: %f, y: %f, z: %f",
      //    tag_frame_id.c_str(), t.transform.translation.x, t.transform.translation.y, t.transform.translation.z);
      //// Create translation and rotation vectors (tVec and rVec) ////
      // copy transform rotation quaternion to a vector. Note that opencv uses w, x, y, z while ROS uses x, y, z, w order
      cv::Vec4d tag_quat_vec(t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z);
      // create the quaternion from the vector
      cv::Quat tag_quat(tag_quat_vec);

      // convert the quaternion to a rotation vector. This is a simplified Rodrigues matrix that opencv uses
      // Append this to the output rotation vector vector
      rVecs.push_back(tag_quat.toRotVec());
      // Create a translation vector and append this to the output translation vector vector
      tVecs.push_back(cv::Vec3d(t.transform.translation.x, t.transform.translation.y, t.transform.translation.z));
  }
  RCLCPP_DEBUG(
          get_logger(), "tVecs size: %ld\n rVecs size: %ld",
          tVecs.size(), rVecs.size());

}

void TFtoAruco(std::string source_frame_id, std::vector<std::string> tag_frame_ids, aruco_opencv_msgs::msg::ArucoDetection& detection)
{
  std::vector<aruco_opencv_msgs::msg::MarkerPose>& marker_vec = detection.markers;
  for( std::string tag_frame_id: tag_frame_ids )
  {
    // Look up for the transformation between tag frame and source frame
      geometry_msgs::msg::TransformStamped t;
      aruco_opencv_msgs::msg::MarkerPose marker;
      try {
        t = tf_buffer_->lookupTransform(
          source_frame_id, tag_frame_id,
          tf2::TimePointZero);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_INFO(
          get_logger(), "Could not transform %s to %s: %s",
          source_frame_id.c_str(), tag_frame_id.c_str(), ex.what());
        return;
      }
      marker.pose.orientation = t.transform.rotation;
      marker.pose.position.x = t.transform.translation.x;
      marker.pose.position.y = t.transform.translation.y;
      marker.pose.position.z = t.transform.translation.z;
      //TODO: Get tag ID from ROS parameters
      // this is just a basic way to add IDs
      //if (tag_frame_id == tag_frame_ids_[0]){marker.marker_id = tag_ids_[0];}
      //else {marker.marker_id = tag_ids_[1];}
      marker.marker_id = tag_id_map_[tag_frame_id];
      marker_vec.push_back(marker);
  }

}

public:
  explicit ArucoTrackerDummy(rclcpp::NodeOptions options)
  : ArucoTracker(options)
  {
    
  }
  LifecycleNodeInterface::CallbackReturn on_activate(const rclcpp_lifecycle::State & state)
  {
    RCLCPP_INFO(get_logger(), "Activating");
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    if (transform_poses_) {
      
      
    }

    LifecycleNode::on_activate(state);

    detection_pub_->on_activate();
    debug_pub_->on_activate();

    on_set_parameter_callback_handle_ =
      add_on_set_parameters_callback(
      std::bind(
        &ArucoTrackerDummy::callback_on_set_parameters,
        this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Waiting for first camera info...");

    cam_info_retrieved_ = false;

    std::string cam_info_topic = image_transport::getCameraInfoTopic(cam_base_topic_);
    cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      cam_info_topic, 1,
      std::bind(&ArucoTrackerDummy::callback_camera_info, this, std::placeholders::_1));

    rmw_qos_profile_t image_sub_qos = rmw_qos_profile_default;
    image_sub_qos.reliability =
      static_cast<rmw_qos_reliability_policy_t>(image_sub_qos_reliability_);
    image_sub_qos.durability = static_cast<rmw_qos_durability_policy_t>(image_sub_qos_durability_);
    image_sub_qos.depth = image_sub_qos_depth_;

    auto qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(image_sub_qos), image_sub_qos);

    img_sub_ = create_subscription<sensor_msgs::msg::Image>(
      cam_base_topic_, qos, std::bind(
        &ArucoTrackerDummy::callback_image, this, std::placeholders::_1));

    return LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }
protected:
  void callback_image(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
    {
      RCLCPP_DEBUG_STREAM(get_logger(), "Image message address [SUBSCRIBE]:\t" << img_msg.get());

      if (!cam_info_retrieved_) {
        return;
      }
      if (img_msg->header.stamp == last_msg_stamp_) {
        RCLCPP_DEBUG(
          get_logger(),
          "The new image has the same timestamp as the previous one (duplicate frame?). Ignoring...");
        return;
      }
      last_msg_stamp_ = img_msg->header.stamp;

      auto callback_start_time = get_clock()->now();

      // Convert the image
      auto cv_ptr = cv_bridge::toCvShare(img_msg);

      std::vector<int> marker_ids;
      std::vector<std::vector<cv::Point2f>> marker_corners;

      // TODO(bjsowa): mutex
      /*
      cv::aruco::detectMarkers(
        cv_ptr->image, dictionary_, marker_corners, marker_ids,
        detector_parameters_);
      */

      cv::Mat rVec(3, 1, cv::DataType<double>::type); // Camera's rotation vector (no rotation, we're in the camera frame)
      rVec.at<double>(0) = 0.0;
      rVec.at<double>(1) = 0.0;
      rVec.at<double>(2) = 0.0;

      cv::Mat tVec(3, 1, cv::DataType<double>::type); // Camera's translation vector (no translation, we're in the camera frame)
      tVec.at<double>(0) = 0.0;
      tVec.at<double>(1) = 0.0;
      tVec.at<double>(2) = 0.0;


      aruco_opencv_msgs::msg::ArucoDetection detection;
      detection.header.frame_id = img_msg->header.frame_id;
      detection.header.stamp = img_msg->header.stamp;
      
      //geometry_msgs::msg::TransformStamped t;
      
      // frame of image to draw on for debug
      //TODO: Get this from the image message (i.e. img_msg->header.frame_id)
      std::string source_frame_id = "camera_color_optical_frame";
      // TF Frame IDs of the virtual tags
      // TODO: Get these from ROS parameters
      //tag_frame_ids_ = {"tag_0_link", "tag_1_link"};
      detection.markers.resize(tag_frame_ids_.size());

      std::vector<cv::Vec3d> tVecs;
      std::vector<cv::Vec3d> rVecs;
      getTagTransforms(source_frame_id, tag_frame_ids_, tVecs, rVecs);
      RCLCPP_DEBUG(
          get_logger(), "tVecs size: %ld\n rVecs size: %ld",
          tVecs.size(), rVecs.size());
      /*
      world_obj_points.push_back(cv::Point3d(t.transform.translation.x, t.transform.translation.y, t.transform.translation.z));
      cv::Vec4d tag_quat_vec(t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z);
      cv::Quat tag_quat(tag_quat_vec);
      cv::Vec3d tag_rvec;
      tag_rvec = tag_quat.toRotVec();

      cv::Vec3d tag_tvec(t.transform.translation.x, t.transform.translation.y, t.transform.translation.z);
      */
      /*
      auto &clk = *this->get_clock();
      RCLCPP_INFO_THROTTLE(
          get_logger(), clk, 100, "Got translation x: %f, y: %f",
          t.transform.translation.x, t.transform.translation.y);
      */
     //world_obj_points.push_back(cv::Point3d(0.01, 0.01, 0.1));

      {
        std::lock_guard<std::mutex> guard(cam_info_mutex_);

        //cv::projectPoints(world_obj_points, rVec, tVec, camera_matrix_, distortion_coeffs_, projectedPoints);
      
      //detection.markers.push_back()
      //detection_pub_->publish(detection);
      TFtoAruco(source_frame_id, tag_frame_ids_, detection);
      detection_pub_->publish(detection);
      
        if (debug_pub_->get_subscription_count() > 0) {
          auto debug_cv_ptr = cv_bridge::toCvCopy(img_msg, "bgr8");

          //cv::circle(debug_cv_ptr->image, projectedPoints[0], 10, cv::Scalar(0, 0, 255), 3);
          // draw tag axes in debug images
          try{
            std::vector<cv::Point3d> world_obj_points;
            if( 1/*(tVecs.size() == rVecs.size())&&(tVecs.size()>0)*/){
            for(int i = 0; i < tVecs.size(); i++)
              {
                cv::drawFrameAxes(debug_cv_ptr->image, camera_matrix_, distortion_coeffs_, rVecs[i], tVecs[i], 0.02, 1);
                //std::vector<cv::Point2f> projectedPoints;
                //world_obj_points[0] = cv::Point3d{detection.markers[i].pose.position.x, detection.markers[i].pose.position.y, detection.markers[i].pose.position.z};
                //cv::projectPoints(world_obj_points, rVec, tVec, camera_matrix_, distortion_coeffs_, projectedPoints);
                //cv::putText(debug_cv_ptr->image, std::to_string(i), projectedPoints[0], cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0,255,0),2,false)
              }
            }
          }
          catch(const cv::Exception& ex){
            RCLCPP_INFO(
              get_logger(), "Could not draw tag axes: %s",
              ex.what());
            return;
          }
          //cv::drawFrameAxes(debug_cv_ptr->image, camera_matrix_, distortion_coeffs_, tag_rvec, tag_tvec, 0.02, 1);
          std::unique_ptr<sensor_msgs::msg::Image> debug_img =
            std::make_unique<sensor_msgs::msg::Image>();
          debug_cv_ptr->toImageMsg(*debug_img);
          debug_pub_->publish(std::move(debug_img));
        }
      }

      auto callback_end_time = get_clock()->now();
      double whole_callback_duration = (callback_end_time - callback_start_time).seconds();
      double image_send_duration = (callback_start_time - img_msg->header.stamp).seconds();

      RCLCPP_DEBUG(
        get_logger(), "Image callback completed. The callback started %.4f s after the image"
        " frame was grabbed and completed its execution in %.4f s.", image_send_duration,
        whole_callback_duration);
      
    }

};

}  // namespace aruco_opencv

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(aruco_opencv::ArucoTracker)
RCLCPP_COMPONENTS_REGISTER_NODE(aruco_opencv::ArucoTrackerAutostart)
RCLCPP_COMPONENTS_REGISTER_NODE(aruco_opencv::ArucoTrackerDummy)