#include <rclcpp/rclcpp.hpp>

#include "LocalPath/Track.h"
#include "pred_opp_traj/detection.h"
#include "pred_opp_traj/collect_detections.h"
#include "pred_opp_traj/gpr_opp_traj.h"

#include "rl_switching_mpc_srv/srv/pred_opp_traj.hpp"

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>


class PredOppTrajServiceNode : public rclcpp::Node
{
public:
    PredOppTrajServiceNode() : Node("pred_opp_traj_service_node")
    {
        this->declare_parameter("map", "");
        this->declare_parameter("horizon", 0);
        this->declare_parameter("dt", 0.0);

        map_ = this->get_parameter("map").as_string();
        horizon_ = this->get_parameter("horizon").as_int();
        dt_ = this->get_parameter("dt").as_double();
        pkg_path_ = ament_index_cpp::get_package_share_directory("pred_opp_traj");

        std::string waypoint_file = pkg_path_ + "/data/path/" + map_ + "_path.csv";
        std::string width_file = pkg_path_ + "/data/path/" + map_ + "_width_info.csv";
        track_ = Track(waypoint_file, width_file);
        curr_time_ = 0.0;

        detection_ = DetectionNode(track_);
        collect_detections_ = CollectDetection(map_, pkg_path_);
        gpr_opp_traj_ = GPROppTrajNode(map_, pkg_path_, horizon_, dt_);

        collect_detections_.init_detections(map_, pkg_path_);

        detected_opp_traj_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/detected_opp_traj_marker", 1);
        visualization_msgs::msg::MarkerArray marker_array;
        for (size_t i = 0; i < collect_detections_.detect_array_.detections.size(); i++)
        {
            pred_msgs::msg::Detection detection = collect_detections_.detect_array_.detections[i];
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = this->now();
            marker.ns = "detected_opp_traj";
            marker.id = i;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose.position.x = detection.x;
            marker.pose.position.y = detection.y;

            tf2::Quaternion q;
            q.setRPY(0, 0, detection.yaw);
            marker.pose.orientation = tf2::toMsg(q);

            marker.scale.x = 0.05;
            marker.scale.y = 0.05;
            marker.scale.z = 1e-5;

            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            marker_array.markers.push_back(marker);
        }
        detected_opp_traj_marker_pub_->publish(marker_array);

        pred_opp_traj_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/pred_opp_traj_marker", 5);
        service_ = this->create_service<rl_switching_mpc_srv::srv::PredOppTraj>(
            "pred_opp_trajectory", std::bind(&PredOppTrajServiceNode::pred_opp_traj_service, this, std::placeholders::_1, std::placeholders::_2));
    };

private:
    std::string map_;
    int horizon_;
    double dt_;
    std::string pkg_path_;
    Track track_;
    double curr_time_;

    pred_msgs::msg::DetectionArray detected_opp_array_;

    DetectionNode detection_;
    CollectDetection collect_detections_;
    GPROppTrajNode gpr_opp_traj_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr detected_opp_traj_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pred_opp_traj_marker_;
    rclcpp::Service<rl_switching_mpc_srv::srv::PredOppTraj>::SharedPtr service_;

    void pred_opp_traj_service(const std::shared_ptr<rl_switching_mpc_srv::srv::PredOppTraj::Request> request, std::shared_ptr<rl_switching_mpc_srv::srv::PredOppTraj::Response> response)
    {
        if (request->reset_collections)
        {
            collect_detections_.init_detections(map_, pkg_path_);
            curr_time_ = 0.0;
        }

        pred_msgs::msg::Detection detected_opp = detection_.detect(request->scan, request->ego_odom, request->opp_odom);
        curr_time_ += 0.025;

        bool is_opp_detected = false;
        if (detected_opp.x_var > 0.0)
            is_opp_detected = true;

        if (is_opp_detected)
        {
            double ego_s = track_.csp.find_s(request->ego_odom.pose.pose.position.x, request->ego_odom.pose.pose.position.y, 0.0);
            double opp_s = track_.csp.find_s(detected_opp.x, detected_opp.y, 0.0);
            if (opp_s - ego_s < -track_.csp.s.back() / 2.0)
                opp_s += track_.csp.s.back();
            else if (opp_s - ego_s > track_.csp.s.back() / 2.0)
                opp_s -= track_.csp.s.back();

            if (std::abs(opp_s - ego_s) > 7.0)
                is_opp_detected = false;
        }

        if (is_opp_detected)
        {
            collect_detections_.add_detection(detected_opp, curr_time_);

            visualization_msgs::msg::MarkerArray marker_array;
            for (size_t i = 0; i < collect_detections_.detect_array_.detections.size(); i++)
            {
                pred_msgs::msg::Detection detection = collect_detections_.detect_array_.detections[i];
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = "map";
                marker.header.stamp = this->now();
                marker.ns = "detected_opp_traj";
                marker.id = i;
                marker.type = visualization_msgs::msg::Marker::SPHERE;
                marker.action = visualization_msgs::msg::Marker::ADD;

                marker.pose.position.x = detection.x;
                marker.pose.position.y = detection.y;

                tf2::Quaternion q;
                q.setRPY(0, 0, detection.yaw);
                marker.pose.orientation = tf2::toMsg(q);

                marker.scale.x = 0.1;
                marker.scale.y = 0.1;
                marker.scale.z = 1e-5;

                marker.color.r = 1.0;
                marker.color.g = 0.0;
                marker.color.b = 0.0;
                marker.color.a = 1.0;

                marker_array.markers.push_back(marker);
            }
            detected_opp_traj_marker_pub_->publish(marker_array);

            double ego_s = track_.csp.find_s(request->ego_odom.pose.pose.position.x, request->ego_odom.pose.pose.position.y, 0.0);
            double opp_s = track_.csp.find_s(detected_opp.x, detected_opp.y, 0.0);
            if (opp_s - ego_s < -track_.csp.s.back() / 2.0)
                opp_s += track_.csp.s.back();
            else if (opp_s - ego_s > track_.csp.s.back() / 2.0)
                opp_s -= track_.csp.s.back();

            if (opp_s - ego_s <= 7.0 && opp_s - ego_s >= -1.0)
            {
                pred_msgs::msg::DetectionArray pred_opp_traj = gpr_opp_traj_.predict_trajectory(detected_opp, collect_detections_.detect_array_);
                response->pred_opp_traj = pred_opp_traj;
            }
            else
                response->pred_opp_traj = pred_msgs::msg::DetectionArray();

        }
        else
            response->pred_opp_traj = pred_msgs::msg::DetectionArray();

        visualization_msgs::msg::MarkerArray markers;
        for (size_t i = 0; i < response->pred_opp_traj.detections.size(); i++) {
            pred_msgs::msg::Detection d = response->pred_opp_traj.detections[i];

            visualization_msgs::msg::Marker m;
            m.header.frame_id = "map";
            m.header.stamp = this->get_clock()->now();
            m.ns = "pred_opp_traj";
            m.id = i;
            m.type = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = d.x;
            m.pose.position.y = d.y;
            m.pose.orientation.w = 1.0;
            m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.01;
            // m.color.r = d.v / 10.0;
            // m.color.g = 0.0;
            // m.color.b = -(d.v - 10.0) / 10.0;
            m.color.g = 1.0;
            m.color.a = 1.0;
            markers.markers.push_back(m);

            m.id = i + 100;
            m.scale.x = d.x_var * 5;
            m.scale.y = d.y_var * 5;
            m.scale.z = 0.0;
            m.color.r = 0.0;
            m.color.g = 0.0;
            m.color.b = 0.0;
            m.color.a = 0.1;
            markers.markers.push_back(m);
        }
        pred_opp_traj_marker_->publish(markers);
    };
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PredOppTrajServiceNode>());
    rclcpp::shutdown();
}