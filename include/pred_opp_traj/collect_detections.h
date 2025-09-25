#ifndef PRED_OPP_TRAJ_CPP__COLLECT_DETECTIONS_H_
#define PRED_OPP_TRAJ_CPP__COLLECT_DETECTIONS_H_

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <string>
#include <cmath>
#include <fstream>
#include <vector>

#include "pred_msgs/msg/detection.hpp"
#include "pred_msgs/msg/detection_array.hpp"

#include "LocalPath/CubicSpline1D.h"
#include "LocalPath/CubicSpline2D.h"

class CollectDetection : public rclcpp::Node
{
public:
    CollectDetection();

private:
    void detection_callback(const pred_msgs::msg::Detection::SharedPtr msg);
    void timer_callback();
    void init_detections();

    std::string map_;

    bool done_init_;
    bool first_point_;

    pred_msgs::msg::DetectionArray detect_array_;
    pred_msgs::msg::Detection prev_detection_;

    double prev_time_;
    int prev_opp_idx_;

    CubicSpline2D sp_;  // center path spline
    CubicSpline1D sp_race_v_; // raceline velocity spline

    rclcpp::Subscription<pred_msgs::msg::Detection>::SharedPtr detection_sub_;
    rclcpp::Publisher<pred_msgs::msg::DetectionArray>::SharedPtr detection_array_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr detected_opp_traj_marker_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

#endif  // PRED_OPP_TRAJ_CPP__COLLECT_DETECTIONS_H_
