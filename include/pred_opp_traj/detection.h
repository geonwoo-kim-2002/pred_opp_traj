#ifndef PRED_OPP_TRAJ_CPP__DETECTION_H_
#define PRED_OPP_TRAJ_CPP__DETECTION_H_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "pred_msgs/msg/detection.hpp"

class DetectionNode : public rclcpp::Node
{
public:
    DetectionNode();

private:
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ego_odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ego_pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr opp_odom_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr opp_box_sub_;

    rclcpp::Publisher<pred_msgs::msg::Detection>::SharedPtr detect_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr detect_marker_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    bool is_simulation_;
    bool is_scan_;
    bool is_ego_odom_;
    bool is_opp_odom_;
    bool is_opp_;

    sensor_msgs::msg::LaserScan scan_;
    nav_msgs::msg::Odometry ego_odom_;
    nav_msgs::msg::Odometry opp_odom_;
    geometry_msgs::msg::PoseStamped ego_pose_;
    vision_msgs::msg::Detection2DArray opp_boxes_;
    double prev_opp_x_;
    double prev_opp_y_;

    void laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void ego_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void ego_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void opp_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void opp_box_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);
    void timer_callback();
};
#endif  // PRED_OPP_TRAJ_CPP__DETECTION_H_