#include "pred_opp_traj/detection.h"

DetectionNode::DetectionNode(Track track)
{
    // this->declare_parameter("is_simulation", true);
    // is_simulation_ = this->get_parameter("is_simulation").as_bool();

    // is_scan_ = false;
    // is_ego_odom_ = false;
    // is_opp_odom_ = false;
    // is_opp_ = false;

    // prev_opp_x_ = 0;
    // prev_opp_y_ = 0;

    // this->declare_parameter("waypoint_file", "");
    // this->declare_parameter("width_file", "");
    // std::string waypoint_file = this->get_parameter("waypoint_file").as_string();
    // std::string width_file = this->get_parameter("width_file").as_string();

    // track_ = Track(waypoint_file, width_file);
    track_ = track;

    // this->declare_parameter("dis_from_wall", 0.0);
    // this->declare_parameter("dis_other", 0.0);
    // this->declare_parameter("dis_static", 0.0);
    // this->declare_parameter("timeout", 0.0);
    // dis_from_wall_ = this->get_parameter("dis_from_wall").as_double();
    // dis_other_ = this->get_parameter("dis_other").as_double();
    // dis_static_ = this->get_parameter("dis_static").as_double();
    // timeout_ = this->get_parameter("timeout").as_double();

    // if (is_simulation_)
    // {
    //     laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>("/scan", 3, std::bind(&DetectionNode::laser_callback, this, std::placeholders::_1));
    //     ego_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/odom", 3, std::bind(&DetectionNode::ego_odom_callback, this, std::placeholders::_1));
    //     opp_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/opp_odom", 3, std::bind(&DetectionNode::opp_odom_callback, this, std::placeholders::_1));
    // }
    // else
    // {
    //     laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>("/scan", rclcpp::QoS(rclcpp::KeepLast(1)), std::bind(&DetectionNode::laser_callback, this, std::placeholders::_1));
    //     ego_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("/mcl_pose", rclcpp::QoS(rclcpp::KeepLast(1)), std::bind(&DetectionNode::ego_pose_callback, this, std::placeholders::_1));
    //     // ego_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/odom", 3, std::bind(&DetectionNode::ego_odom_callback, this, std::placeholders::_1));
    //     opp_box_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>("/bounding_box", rclcpp::QoS(rclcpp::KeepLast(1)), std::bind(&DetectionNode::opp_box_callback, this, std::placeholders::_1));
    // }

    // detect_pub_ = this->create_publisher<pred_msgs::msg::Detection>("/detection", rclcpp::QoS(rclcpp::KeepLast(1)));
    // detect_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/detection_marker", rclcpp::QoS(rclcpp::KeepLast(1)));

    // timer_ = this->create_wall_timer(std::chrono::milliseconds(25), std::bind(&DetectionNode::timer_callback, this));
}

pred_msgs::msg::Detection DetectionNode::detect(const sensor_msgs::msg::LaserScan scan, const nav_msgs::msg::Odometry ego_odom, const nav_msgs::msg::Odometry opp_odom)
{
    double ego_x = ego_odom.pose.pose.position.x;
    double ego_y = ego_odom.pose.pose.position.y;
    // auto ego_quat = ego_odom.pose.pose.orientation;
    double ego_yaw = tf2::getYaw(ego_odom.pose.pose.orientation);

    double opp_x = opp_odom.pose.pose.position.x;
    double opp_y = opp_odom.pose.pose.position.y;
    // auto opp_quat = opp_odom.pose.pose.orientation;
    double opp_yaw = tf2::getYaw(opp_odom.pose.pose.orientation);

    double closest = 100.0;
    pred_msgs::msg::Detection detect_opp = pred_msgs::msg::Detection();
    // std::cout << "init opp" << detect_opp.x << ", " << detect_opp.y << std::endl;
    for (size_t i = 0; i < scan.ranges.size(); i++)
    {
        double angle = scan.angle_min + i * scan.angle_increment;
        double scan_range = scan.ranges[i];
        double scan_x = ego_x + scan_range * cos(ego_yaw + angle);
        double scan_y = ego_y + scan_range * sin(ego_yaw + angle);

        double distance = hypot(scan_x - opp_x, scan_y - opp_y);
        if (distance < closest)
            closest = distance;

        if (closest < 0.3)
        {
            detect_opp.dt = 0.0;
            detect_opp.x = opp_x;
            detect_opp.y = opp_y;

            // tf2::Quaternion q_opp(opp_quat.x, opp_quat.y, opp_quat.z, opp_quat.w);
            // double roll_o, pitch_o, yaw_o;
            // tf2::Matrix3x3(q_opp).getRPY(roll_o, pitch_o, yaw_o);
            detect_opp.yaw = opp_yaw;
            detect_opp.v = hypot(opp_odom.twist.twist.linear.x, opp_odom.twist.twist.linear.y);

            detect_opp.x_var = 0.001;
            detect_opp.y_var = 0.001;
            detect_opp.yaw_var = 0.05;
            detect_opp.v_var = 0.05;

            break;
        }
    }
    return detect_opp;
}

// void DetectionNode::laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
// {
//     is_scan_ = true;
//     scan = *msg;
// }

// void DetectionNode::ego_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
// {
//     is_ego_odom_ = true;
//     ego_odom_ = *msg;

//     ego_pose_.pose = msg->pose.pose;
//     ego_pose_.header = msg->header;
// }

// void DetectionNode::ego_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
// {
//     is_ego_odom_ = true;
//     ego_pose_ = *msg;
// }

// void DetectionNode::opp_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
// {
//     is_opp_odom_ = true;
//     opp_odom_ = *msg;
// }

// void DetectionNode::opp_box_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
// {
//     is_opp_ = true;
//     opp_boxes_ = *msg;
// }

// void DetectionNode::timer_callback()
// {
//     if (is_simulation_)
//     {
//         if (is_scan_ && is_ego_odom_ && is_opp_odom_)
//         {
//         double ego_x = ego_odom_.pose.pose.position.x;
//         double ego_y = ego_odom_.pose.pose.position.y;
//         // auto ego_quat = ego_odom_.pose.pose.orientation;
//         double opp_x = opp_odom_.pose.pose.position.x;
//         double opp_y = opp_odom_.pose.pose.position.y;
//         // auto opp_quat = opp_odom_.pose.pose.orientation;

//         // tf2::Quaternion q_ego(ego_quat.x, ego_quat.y, ego_quat.z, ego_quat.w);
//         // double roll, pitch, yaw;
//         // tf2::Matrix3x3(q_ego).getRPY(roll, pitch, yaw);
//         double yaw = tf2::getYaw(ego_odom_.pose.pose.orientation);
//         double opp_yaw = tf2::getYaw(opp_odom_.pose.pose.orientation);

//         double closest = 100.0;
//         for (size_t i = 0; i < scan.ranges.size(); i++)
//         {
//             double angle = scan.angle_min + i * scan_.angle_increment;
//             double scan_range = scan_.ranges[i];
//             double scan_x = ego_x + scan_range * cos(yaw + angle);
//             double scan_y = ego_y + scan_range * sin(yaw + angle);

//             double distance = hypot(scan_x - opp_x, scan_y - opp_y);
//             if (distance < closest)
//                 closest = distance;

//             if (closest < 0.3)
//             {
//                 auto detection_msg = pred_msgs::msg::Detection();
//                 detection_msg.dt = 0.0;
//                 detection_msg.x = opp_x;
//                 detection_msg.y = opp_y;

//                 // tf2::Quaternion q_opp(opp_quat.x, opp_quat.y, opp_quat.z, opp_quat.w);
//                 // double roll_o, pitch_o, yaw_o;
//                 // tf2::Matrix3x3(q_opp).getRPY(roll_o, pitch_o, yaw_o);
//                 detection_msg.yaw = opp_yaw;
//                 detection_msg.v = hypot(opp_odom_.twist.twist.linear.x, opp_odom_.twist.twist.linear.y);

//                 detection_msg.x_var = 0.05;
//                 detection_msg.y_var = 0.05;
//                 detection_msg.yaw_var = 0.05;
//                 detection_msg.v_var = 0.05;

//                 detect_pub_->publish(detection_msg);

//                 visualization_msgs::msg::Marker marker = visualization_msgs::msg::Marker();
//                 marker.header.frame_id = "map";
//                 marker.header.stamp = this->get_clock()->now();
//                 marker.id = 0;
//                 marker.type = visualization_msgs::msg::Marker::ARROW;
//                 marker.action = visualization_msgs::msg::Marker::ADD;
//                 marker.pose.position.x = opp_x;
//                 marker.pose.position.y = opp_y;
//                 marker.pose.orientation = opp_odom_.pose.pose.orientation;
//                 marker.scale.x = detection_msg.v * 0.2;
//                 marker.scale.y = 0.2;
//                 marker.scale.z = 0.2;
//                 marker.color.r = 1.0;
//                 marker.color.g = 0.0;
//                 marker.color.b = 0.0;
//                 marker.color.a = 1.0;
//                 detect_marker_pub_->publish(marker);

//                 break;
//             }
//         }
//         }
//     }
//     else
//     {
//         static int count = 0;
//         static double prev_time = 0.0;
//         static bool first_detection = true;
//         static bool opp_is_static = true;
//         if (is_scan_ && is_ego_odom_ && opp_boxes_.detections.size() > 0)
//         {
//             double ego_x = ego_pose_.pose.position.x;
//             double ego_y = ego_pose_.pose.position.y;
//             // auto ego_quat = ego_pose_.pose.orientation;

//             // tf2::Quaternion q_ego(ego_quat.x, ego_quat.y, ego_quat.z, ego_quat.w);
//             // double roll, pitch, yaw;
//             // tf2::Matrix3x3(q_ego).getRPY(roll, pitch, yaw);
//             double yaw = tf2::getYaw(ego_pose_.pose.orientation);

//             double opp_local_x = opp_boxes_.detections[0].bbox.center.position.x;
//             double opp_local_y = opp_boxes_.detections[0].bbox.center.position.y;

//             double opp_x = ego_x + opp_local_x * cos(yaw) - opp_local_y * sin(yaw);
//             double opp_y = ego_y + opp_local_x * sin(yaw) + opp_local_y * cos(yaw);
//             if (std::hypot(ego_x - opp_x, ego_y - opp_y) > 8.0)
//                 return;

//             double opp_s = track_.csp.find_s(opp_x, opp_y, 0.0);
//             std::cout << "opp s: " << opp_s << std::endl;
//             int opp_width_idx = round(opp_s * 100);
//             double left_width = track_.lane[opp_width_idx].left_width - dis_from_wall_;
//             double right_width = track_.lane[opp_width_idx].right_width - dis_from_wall_;
//             double opp_lat_dis = track_.csp.calc_lateral_deviation(opp_x, opp_y, opp_s);
//             if ((opp_lat_dis < 0 && -opp_lat_dis > left_width) || (opp_lat_dis >= 0 && opp_lat_dis > right_width))
//                 return;
//             std::cout << "opp lat dis: " << opp_lat_dis << std::endl;

//             double dis = std::hypot(prev_opp_x_ - opp_x, prev_opp_y_ - opp_y);

//             if (dis > dis_other_)
//             {
//                 first_detection = true;
//                 opp_is_static = true;
//                 count = 0;
//             }
//             else if (count == 0 && first_detection == true)
//             {
//                 prev_opp_x_ = opp_x;
//                 prev_opp_y_ = opp_y;
//                 first_detection = false;
//             }
//             else if ((dis > dis_static_ && first_detection == false))
//             {
//                 opp_is_static = false;
//                 prev_opp_x_ = opp_x;
//                 prev_opp_y_ = opp_y;
//                 count = 0;
//             }
//             else if (first_detection == false)
//             {
//                 count++;
//                 // opp_v = 0.0;
//                 if (count > 5)
//                 {
//                     // opp_v = 0.0;
//                     opp_is_static = true;
//                 }
//             }
            
//             prev_time = this->get_clock()->now().seconds();

//             pred_msgs::msg::Detection detection_msg = pred_msgs::msg::Detection();
//             detection_msg.dt = 0.0;
//             detection_msg.x = opp_x;
//             detection_msg.y = opp_y;
//             detection_msg.yaw = 0.0;
//             if (opp_is_static)
//                 detection_msg.v = 0.0;
//             else
//                 detection_msg.v = 3.0;
//             // if (dis < 0.03)
//             // {
//             //     detection_msg.v = -1.0;
//             // }
//             // std::cout << "detection dis: " << dis << std::endl;
//             detection_msg.x_var = 0.05;
//             detection_msg.y_var = 0.05;
//             detection_msg.yaw_var = 0.05;
//             detection_msg.v_var = 0.05;
//             detect_pub_->publish(detection_msg);

//             visualization_msgs::msg::Marker marker = visualization_msgs::msg::Marker();
//             marker.header.frame_id = "map";
//             marker.header.stamp = this->get_clock()->now();
//             marker.id = 0;
//             marker.type = visualization_msgs::msg::Marker::SPHERE;
//             marker.action = visualization_msgs::msg::Marker::ADD;
//             marker.pose.position.x = opp_x;
//             marker.pose.position.y = opp_y;
//             marker.pose.orientation.w = 1.0;
//             marker.scale.x = 0.2;
//             marker.scale.y = 0.2;
//             marker.scale.z = 1e-5;
//             marker.color.r = 1.0;
//             marker.color.g = 0.0;
//             marker.color.b = 0.0;
//             marker.color.a = 1.0;
//             detect_marker_pub_->publish(marker);

//             is_opp_ = false;
//         }

//         if (this->get_clock()->now().seconds() - prev_time > timeout_)
//         {
//             first_detection = true;
//             opp_is_static = true;
//             count = 0;
//             std::cout << "time reset" << std::endl;
//         }
//     }
// }

// int main(int argc, char **argv)
// {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<DetectionNode>();
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }
