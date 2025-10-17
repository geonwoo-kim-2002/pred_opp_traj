#include "pred_opp_traj/detection.h"

DetectionNode::DetectionNode() : Node("detection_node")
{
    this->declare_parameter("is_simulation", true);
    is_simulation_ = this->get_parameter("is_simulation").as_bool();

    is_scan_ = false;
    is_ego_odom_ = false;
    is_opp_odom_ = false;
    is_opp_ = false;

    prev_opp_x_ = 0;
    prev_opp_y_ = 0;

    if (is_simulation_)
    {
        laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>("/scan", 3, std::bind(&DetectionNode::laser_callback, this, std::placeholders::_1));
        ego_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/odom", 3, std::bind(&DetectionNode::ego_odom_callback, this, std::placeholders::_1));
        opp_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/opp_odom", 3, std::bind(&DetectionNode::opp_odom_callback, this, std::placeholders::_1));
    }
    else
    {
        laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>("/scan", rclcpp::QoS(rclcpp::KeepLast(1)), std::bind(&DetectionNode::laser_callback, this, std::placeholders::_1));
        ego_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("/mcl_pose", rclcpp::QoS(rclcpp::KeepLast(1)), std::bind(&DetectionNode::ego_pose_callback, this, std::placeholders::_1));
        // ego_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/odom", 3, std::bind(&DetectionNode::ego_odom_callback, this, std::placeholders::_1));
        opp_box_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>("/bounding_box", rclcpp::QoS(rclcpp::KeepLast(1)), std::bind(&DetectionNode::opp_box_callback, this, std::placeholders::_1));
    }

    detect_pub_ = this->create_publisher<pred_msgs::msg::Detection>("/detection", rclcpp::QoS(rclcpp::KeepLast(1)));
    detect_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/detection_marker", rclcpp::QoS(rclcpp::KeepLast(1)));

    timer_ = this->create_wall_timer(std::chrono::milliseconds(25), std::bind(&DetectionNode::timer_callback, this));
}

void DetectionNode::laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    is_scan_ = true;
    scan_ = *msg;
}

void DetectionNode::ego_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    is_ego_odom_ = true;
    ego_odom_ = *msg;

    ego_pose_.pose = msg->pose.pose;
    ego_pose_.header = msg->header;
}

void DetectionNode::ego_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    is_ego_odom_ = true;
    ego_pose_ = *msg;
}

void DetectionNode::opp_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    is_opp_odom_ = true;
    opp_odom_ = *msg;
}

void DetectionNode::opp_box_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
    is_opp_ = true;
    opp_boxes_ = *msg;
}

void DetectionNode::timer_callback()
{
    if (is_simulation_)
    {
        if (is_scan_ && is_ego_odom_ && is_opp_odom_)
        {
        double ego_x = ego_odom_.pose.pose.position.x;
        double ego_y = ego_odom_.pose.pose.position.y;
        // auto ego_quat = ego_odom_.pose.pose.orientation;
        double opp_x = opp_odom_.pose.pose.position.x;
        double opp_y = opp_odom_.pose.pose.position.y;
        // auto opp_quat = opp_odom_.pose.pose.orientation;

        // tf2::Quaternion q_ego(ego_quat.x, ego_quat.y, ego_quat.z, ego_quat.w);
        // double roll, pitch, yaw;
        // tf2::Matrix3x3(q_ego).getRPY(roll, pitch, yaw);
        double yaw = tf2::getYaw(ego_odom_.pose.pose.orientation);
        double opp_yaw = tf2::getYaw(opp_odom_.pose.pose.orientation);

        double closest = 100.0;
        for (size_t i = 0; i < scan_.ranges.size(); i++)
        {
            double angle = scan_.angle_min + i * scan_.angle_increment;
            double scan_range = scan_.ranges[i];
            double scan_x = ego_x + scan_range * cos(yaw + angle);
            double scan_y = ego_y + scan_range * sin(yaw + angle);

            double distance = hypot(scan_x - opp_x, scan_y - opp_y);
            if (distance < closest)
                closest = distance;

            if (closest < 0.3)
            {
                auto detection_msg = pred_msgs::msg::Detection();
                detection_msg.dt = 0.0;
                detection_msg.x = opp_x;
                detection_msg.y = opp_y;

                // tf2::Quaternion q_opp(opp_quat.x, opp_quat.y, opp_quat.z, opp_quat.w);
                // double roll_o, pitch_o, yaw_o;
                // tf2::Matrix3x3(q_opp).getRPY(roll_o, pitch_o, yaw_o);
                detection_msg.yaw = opp_yaw;
                detection_msg.v = hypot(opp_odom_.twist.twist.linear.x, opp_odom_.twist.twist.linear.y);

                detection_msg.x_var = 0.05;
                detection_msg.y_var = 0.05;
                detection_msg.yaw_var = 0.05;
                detection_msg.v_var = 0.05;

                detect_pub_->publish(detection_msg);

                visualization_msgs::msg::Marker marker = visualization_msgs::msg::Marker();
                marker.header.frame_id = "map";
                marker.header.stamp = this->get_clock()->now();
                marker.id = 0;
                marker.type = visualization_msgs::msg::Marker::ARROW;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.pose.position.x = opp_x;
                marker.pose.position.y = opp_y;
                marker.pose.orientation = opp_odom_.pose.pose.orientation;
                marker.scale.x = detection_msg.v * 0.2;
                marker.scale.y = 0.2;
                marker.scale.z = 0.2;
                marker.color.r = 1.0;
                marker.color.g = 0.0;
                marker.color.b = 0.0;
                marker.color.a = 1.0;
                detect_marker_pub_->publish(marker);

                break;
            }
        }
        }
    }
    else
    {
        if (is_scan_ && is_ego_odom_ && opp_boxes_.detections.size() > 0)
        {
            double ego_x = ego_pose_.pose.position.x;
            double ego_y = ego_pose_.pose.position.y;
            // auto ego_quat = ego_pose_.pose.orientation;

            // tf2::Quaternion q_ego(ego_quat.x, ego_quat.y, ego_quat.z, ego_quat.w);
            // double roll, pitch, yaw;
            // tf2::Matrix3x3(q_ego).getRPY(roll, pitch, yaw);
            double yaw = tf2::getYaw(ego_pose_.pose.orientation);

            double opp_local_x = opp_boxes_.detections[0].bbox.center.position.x;
            double opp_local_y = opp_boxes_.detections[0].bbox.center.position.y;

            double opp_x = ego_x + opp_local_x * cos(yaw) - opp_local_y * sin(yaw);
            double opp_y = ego_y + opp_local_x * sin(yaw) + opp_local_y * cos(yaw);

            double dis = std::hypot(prev_opp_x_ - opp_x, prev_opp_y_ - opp_y);

            pred_msgs::msg::Detection detection_msg = pred_msgs::msg::Detection();
            detection_msg.dt = 0.0;
            detection_msg.x = opp_x;
            detection_msg.y = opp_y;
            detection_msg.yaw = 0.0;
            detection_msg.v = 0.0;
            // if (dis < 0.03)
            // {
            //     detection_msg.v = -1.0;
            // }
            // std::cout << "detection dis: " << dis << std::endl;
            detection_msg.x_var = 0.05;
            detection_msg.y_var = 0.05;
            detection_msg.yaw_var = 0.05;
            detection_msg.v_var = 0.05;
            detect_pub_->publish(detection_msg);

            visualization_msgs::msg::Marker marker = visualization_msgs::msg::Marker();
            marker.header.frame_id = "map";
            marker.header.stamp = this->get_clock()->now();
            marker.id = 0;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = opp_x;
            marker.pose.position.y = opp_y;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 1e-5;
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;
            detect_marker_pub_->publish(marker);

            prev_opp_x_ = opp_x;
            prev_opp_y_ = opp_y;

            is_opp_ = false;
        }
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
