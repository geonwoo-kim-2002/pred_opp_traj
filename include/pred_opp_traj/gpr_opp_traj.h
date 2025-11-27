#ifndef PRED_OPP_TRAJ__GPR_OPP_TRAJ_H_
#define PRED_OPP_TRAJ__GPR_OPP_TRAJ_H_

// #include <rclcpp/rclcpp.hpp>
// #include <visualization_msgs/msg/marker_array.hpp>
// #include <ament_index_cpp/get_package_share_directory.hpp>

#include <cmath>
#include <vector>
#include <fstream>
#include <Eigen/Dense>

#include "gp.h"
#include "gp_utils.h"
#include "cov_factory.h"

#include "pred_msgs/msg/detection.hpp"
#include "pred_msgs/msg/detection_array.hpp"

#include "LocalPath/CubicSpline2D.h"

struct GPPrediction
{
    std::vector<double> mean;
    std::vector<double> std_dev;
};

class GPROppTrajNode
{
public:
    GPROppTrajNode() {};
    GPROppTrajNode(std::string map, std::string pkg_path, int horizon, double dt);
    pred_msgs::msg::DetectionArray predict_trajectory(const pred_msgs::msg::Detection curr_opp, const pred_msgs::msg::DetectionArray det_arr);

private:
    // rclcpp::Subscription<pred_msgs::msg::Detection>::SharedPtr det_sub_;
    // rclcpp::Subscription<pred_msgs::msg::DetectionArray>::SharedPtr det_arr_sub_;
    // rclcpp::Publisher<pred_msgs::msg::DetectionArray>::SharedPtr pred_pub_;
    // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    // rclcpp::TimerBase::SharedPtr timer_;

    pred_msgs::msg::Detection curr_opp_;
    pred_msgs::msg::DetectionArray det_arr_;
    bool is_curr_opp_;
    bool is_det_arr_;

    int horizon_;
    double dt_;

    CubicSpline2D sp_;

    // void detection_callback(const pred_msgs::msg::Detection::SharedPtr msg);
    // void detection_array_callback(const pred_msgs::msg::DetectionArray::SharedPtr msg);
    // void timer_callback();
};

#endif  // PRED_OPP_TRAJ__GPR_OPP_TRAJ_H_