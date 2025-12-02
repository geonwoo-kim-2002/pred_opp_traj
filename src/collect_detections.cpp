#include "pred_opp_traj/collect_detections.h"

CollectDetection::CollectDetection(std::string map, std::string pkg_path):
// : Node("collect_detection_node"),
//   done_init_(false),
  first_point_(true),
  prev_time_(0.0),
  prev_opp_idx_(-1)
{
    std::string center_csv = pkg_path + "/data/path/" + map + "_path.csv";
    std::ifstream path_file;
    path_file.open(center_csv);
    if (!path_file.is_open())
    {
        // RCLCPP_ERROR(this->get_logger(), "Failed to load center path CSV");
        std::cout << "Failed to load center path CSV" << std::endl;
        return;
    }
    else
    {
        std::vector<f1_msgs::msg::Waypoint> center_path;
        std::string line;
        bool first_line = true;
        while (std::getline(path_file, line))
        {
            std::stringstream ss(line);
            std::string value;
            if (first_line)
            {
                first_line = false;
                continue; // skip header
            }
            std::vector<float> row;
            while (std::getline(ss, value, ','))
                row.push_back(std::stof(value));

            if (row.size() >= 2)
            {
                f1_msgs::msg::Waypoint pos;
                pos.x = row[0];
                pos.y = row[1];
                pos.v = 0;
                pos.vm = 0;
                center_path.push_back(pos);
            }
        }
        path_file.close();
        sp_.setCubicSpline2D(center_path);
    }
    // this->declare_parameter<std::string>("map", "");
    // map_ = this->get_parameter("map").as_string();

    // init_detections();

    // detection_sub_ = this->create_subscription<pred_msgs::msg::Detection>("/detection", 1, std::bind(&CollectDetection::detection_callback, this, std::placeholders::_1));

    // detection_array_pub_ = this->create_publisher<pred_msgs::msg::DetectionArray>("/detected_opp_traj", 1);
    // detected_opp_traj_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/detected_opp_traj_marker", 1);

    // timer_ = this->create_wall_timer(std::chrono::milliseconds(5), std::bind(&CollectDetection::timer_callback, this));
}

void CollectDetection::init_detections(std::string map, std::string pkg_path)
{
    // std::string pkg_path = ament_index_cpp::get_package_share_directory("pred_opp_traj");
    prev_time_ = 0.0;
    prev_opp_idx_ = -1;

    std::string race_csv = pkg_path + "/data/raceline/" + map + "_race_spline.csv";
    std::ifstream race_file;
    race_file.open(race_csv);
    detect_array_.detections.clear();
    if (race_file.is_open())
    {
        // RCLCPP_INFO(this->get_logger(), "Using precomputed raceline spline data. %s", race_csv.c_str());
        std::cout << "Using precomputed raceline spline data. " << race_csv << std::endl;
        std::string line;
        bool first_line = true;
        while (std::getline(race_file, line))
        {
            std::stringstream ss(line);
            std::string value;
            if (first_line)
            {
                first_line = false;
                continue;
            }
            std::vector<float> row;
            while (std::getline(ss, value, ','))
                row.push_back(std::stof(value));

            if (row.size() >= 9)
            {
                pred_msgs::msg::Detection detection;
                detection.dt = row[0];
                detection.x = row[1];
                detection.y = row[2];
                detection.yaw = row[3];
                detection.v = row[4];
                detection.x_var = row[5];
                detection.y_var = row[6];
                detection.yaw_var = row[7];
                detection.v_var = row[8];
                detect_array_.detections.push_back(detection);
            }
        }
        race_file.close();
    }
    else
    {
        // RCLCPP_WARN(this->get_logger(), "Raceline CSV not found, computing raceline...");
        std::cout << "Raceline CSV not found, computing raceline..." << std::endl;

        vector<double> center_x, center_y;
        for (double i = 0.0; i < sp_.s.back(); i += 0.1)
        {
            center_x.push_back(sp_.calc_x(i));
            center_y.push_back(sp_.calc_y(i));
        }

        std::vector<double> states_s, t_s;
        std::string states_csv = pkg_path + "/data/raceline/" + map + "_states.csv";
        std::ifstream states_file;
        states_file.open(states_csv);
        std::string line;
        bool first_line = true;
        while (std::getline(states_file, line))
        {
            std::stringstream ss(line);
            std::string value;
            if (first_line)
            {
                first_line = false;
                continue;
            }
            std::vector<double> row;
            while (std::getline(ss, value, ','))
                row.push_back(std::stod(value));
            if (row.size() >= 2)
            {
                states_s.push_back(row[0]);
                t_s.push_back(row[1]);
            }
        }
        states_file.close();

        if (center_x.size() != t_s.size())
        {
            std::vector<double> t_s_interp;
            for (double s = 0.0; s < sp_.s.back(); s += 0.1)
            {
                std::vector<double>::iterator it = std::lower_bound(states_s.begin(), states_s.end(), s);
                if (it == states_s.begin())
                    t_s_interp.push_back(t_s.front());
                else if (it == states_s.end())
                    t_s_interp.push_back(t_s_interp.back() + (t_s_interp.back() - t_s_interp[t_s_interp.size() - 2]));
                else
                {
                    int idx = it - states_s.begin();
                    double s0 = states_s[idx - 1], s1 = states_s[idx];
                    double t0 = t_s[idx - 1], t1 = t_s[idx];
                    double interp = t0 + (s - s0) / (s1 - s0) * (t1 - t0);
                    t_s_interp.push_back(interp);
                }
            }
            t_s = t_s_interp;
        }
        // RCLCPP_INFO(this->get_logger(), "Length of center path: %d, Length of states data: %d\n%f", (int)center_x.size(), (int)t_s.size(), t_s.back());
        std::cout << "Length of center path: " << (int)center_x.size() << ", Length of states data: " << (int)t_s.size() << "\n" << t_s.back() << std::endl;

        std::vector<f1_msgs::msg::Waypoint> race_path;
        std::vector<double> race_v;
        std::string race_path_csv = pkg_path + "/data/raceline/" + map + "_traj_race_cl.csv";
        std::ifstream race_path_file;
        race_path_file.open(race_path_csv);
        std::string line2;
        first_line = true;
        while (std::getline(race_path_file, line2))
        {
            std::stringstream ss(line2);
            std::string value;
            if (first_line)
            {
                first_line = false;
                continue; // skip header
            }
            std::vector<float> row;
            while (std::getline(ss, value, ','))
                row.push_back(std::stof(value));

            if (row.size() >= 2)
            {
                f1_msgs::msg::Waypoint pos;
                pos.x = row[1];
                pos.y = row[2];
                pos.v = 0;
                pos.vm = 0;
                race_path.push_back(pos);

                race_v.push_back(row[5]);
            }
        }
        race_path_file.close();

        CubicSpline2D race_spline(race_path);
        CubicSpline1D race_v_spline(race_spline.s, race_v);

        std::vector<double> race_x, race_y, race_yaw;
        race_v.clear();
        for (double s = 0.0; s < race_spline.s.back(); s += 0.01)
        {
            race_x.push_back(race_spline.calc_x(s));
            race_y.push_back(race_spline.calc_y(s));
            race_yaw.push_back(race_spline.calc_yaw(s));
            race_v.push_back(race_v_spline.calc_der0(s));
        }

        detect_array_ = pred_msgs::msg::DetectionArray();
        for (size_t i = 0; i < center_x.size(); i++)
        {
            double min_dist = 1e3;
            int min_idx = -1;
            for (size_t j = 0; j < race_x.size(); j++)
            {
                double dist = std::hypot(center_x[i] - race_x[j], center_y[i] - race_y[j]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_idx = j;
                }
            }

            pred_msgs::msg::Detection d;
            if (i == 0)
                d.dt = t_s[t_s.size() - 1] - t_s[t_s.size() - 2];
            else
                d.dt = t_s[i] - t_s[i - 1];
            d.x = race_x[min_idx];
            d.y = race_y[min_idx];
            d.yaw = race_yaw[min_idx];
            d.v = race_v[min_idx];
            d.x_var = 1.0;
            d.y_var = 1.0;
            d.yaw_var = 1.0;
            d.v_var = 1.0;
            detect_array_.detections.push_back(d);
        }

        std::string race_spline_csv = pkg_path + "/data/raceline/" + map + "_race_spline.csv";
        std::ofstream file(race_spline_csv);
        file << "dt,x,y,yaw,v,x_var,y_var,yaw_var,v_var\n";
        for (int i = 0; i < (int)detect_array_.detections.size(); i++)
        {
            pred_msgs::msg::Detection d = detect_array_.detections[i];
            file << d.dt << "," << d.x << "," << d.y << "," << d.yaw << "," << d.v << ","
                 << d.x_var << "," << d.y_var << "," << d.yaw_var << "," << d.v_var << "\n";
        }
        // RCLCPP_INFO(this->get_logger(), "Save raceline spline: %s", race_spline_csv.c_str());
        std::cout << "Save raceline spline: " << race_spline_csv << std::endl;
    }

    done_init_ = true;
    // RCLCPP_INFO(this->get_logger(), "Detections initialized for map: %s", map.c_str());
    std::cout << "Detections initialized for map: " << map << std::endl;
}

void CollectDetection::add_detection(pred_msgs::msg::Detection detected_opp, double curr_time)
{
    // double dis = std::hypot(detected_opp.x - prev_detection_.x, detected_opp.y - prev_detection_.y);
    if (detected_opp.v < 1.0)
        return;

    if (curr_time - prev_time_ >= 0.1)
    {
        first_point_ = true;
        // prev_detection_ = detected_opp;
    }

    if (detected_opp != prev_detection_)
    {
        double curr_opp_s = sp_.find_s(detected_opp.x, detected_opp.y, 0.0);
        // std::cout << "x: " << detected_opp.x << ", y: " << detected_opp.y << ", curr_opp_s: " << curr_opp_s << std::endl;
        if ((int)std::round(curr_opp_s * 100) % 10 <= 3 || (int)std::round(curr_opp_s * 100) % 10 >= 7)
        {
            int opp_idx = std::round(curr_opp_s * 10);
            if (opp_idx >= (int)detect_array_.detections.size())
                opp_idx -= detect_array_.detections.size();

            if (first_point_)
            {
                first_point_ = false;
                prev_time_ = curr_time;
                prev_opp_idx_ = opp_idx;
            }
            else
            {
                double dt = curr_time - prev_time_;
                std::cout << "opp s: " << curr_opp_s << ", opp idx: " << opp_idx << ", dt: " << dt << ", x: " << detected_opp.x << ", y: " << detected_opp.y << std::endl;

                if (opp_idx - prev_opp_idx_ < -(int)detect_array_.detections.size() / 2)
                {
                    for (int i = prev_opp_idx_; i < opp_idx + (int)detect_array_.detections.size(); i++)
                        detect_array_.detections[(i + 1) % detect_array_.detections.size()].dt = dt / (opp_idx + detect_array_.detections.size() - prev_opp_idx_);
                }
                else
                {
                    for (int i = prev_opp_idx_; i < opp_idx; i++)
                        detect_array_.detections[(i + 1)].dt = dt / (opp_idx - prev_opp_idx_);
                }

                prev_opp_idx_ = opp_idx;
                prev_time_ = curr_time;

                pred_msgs::msg::Detection detection = detected_opp;
                detection.dt = detect_array_.detections[opp_idx].dt;
                detection.v = detect_array_.detections[opp_idx].v;

                if (std::hypot(detection.x - detect_array_.detections[opp_idx].x, detection.y - detect_array_.detections[opp_idx].y) > 0.4)
                {
                    detection.x_var = 0.5;
                    detection.y_var = 0.5;
                    detection.yaw_var = 0.5;
                    detection.v_var = 0.5;
                }
                detect_array_.detections[opp_idx] = detection;
            }

            prev_detection_ = detected_opp;
        }
    }
}

// void CollectDetection::detection_callback(const pred_msgs::msg::Detection::SharedPtr msg)
// {
//     double now = this->get_clock()->now().seconds();
//     double dis = std::hypot(msg->x - prev_detection_.x, msg->y - prev_detection_.y);
//     // std::cout << "dis: " << dis << std::endl;
//     if (msg->v < 1.0)
//         return;


//     // if (dis > 0.25) {
//     //     prev_detection_ = *msg;
//     //     return;
//     // } else if (dis < 0.15) {
//     //     // std::cout << "dis: " << dis << std::endl;
//     //     return;
//     // }

//     if (now - prev_time_ >= 0.5)
//     {
//         first_point_ = true;
//         // prev_detection_ = *msg;
//     }

//     if (*msg != prev_detection_ && done_init_)
//     {
//         double curr_opp_s = sp_.find_s(msg->x, msg->y, 0.0);
//         if ((int)std::round(curr_opp_s * 100) % 10 <= 2 || (int)std::round(curr_opp_s * 100) % 10 >= 8)
//         {
//             int opp_idx = std::round(curr_opp_s * 10);
//             if (opp_idx >= (int)detect_array_.detections.size())
//                 opp_idx -= detect_array_.detections.size();

//             if (first_point_)
//             {
//                 first_point_ = false;
//                 prev_time_ = now;
//                 prev_opp_idx_ = opp_idx;
//             }
//             else
//             {
//                 double dt = now - prev_time_;

//                 if (opp_idx - prev_opp_idx_ < -(int)detect_array_.detections.size() / 2)
//                 {
//                     for (int i = prev_opp_idx_; i < opp_idx + (int)detect_array_.detections.size(); i++)
//                         detect_array_.detections[(i + 1) % detect_array_.detections.size()].dt = dt / (opp_idx + detect_array_.detections.size() - prev_opp_idx_);
//                 }
//                 else
//                 {
//                     for (int i = prev_opp_idx_; i < opp_idx; i++)
//                         detect_array_.detections[(i + 1)].dt = dt / (opp_idx - prev_opp_idx_);
//                 }

//                 prev_opp_idx_ = opp_idx;
//                 prev_time_ = now;

//                 pred_msgs::msg::Detection detection = *msg;
//                 detection.dt = detect_array_.detections[opp_idx].dt;
//                 detection.v = detect_array_.detections[opp_idx].v;
//                 detect_array_.detections[opp_idx] = detection;
//             }

//             prev_detection_ = *msg;
//         }
//     }
// }

// void CollectDetection::timer_callback()
// {
//     if (!done_init_)
//         return;

//     visualization_msgs::msg::MarkerArray marker_array;
//     for (size_t i = 0; i < detect_array_.detections.size(); i++)
//     {
//         auto detection = detect_array_.detections[i];
//         visualization_msgs::msg::Marker marker;
//         marker.header.frame_id = "map";
//         marker.header.stamp = this->now();
//         marker.ns = "detected_opp_traj";
//         marker.id = i;
//         marker.type = visualization_msgs::msg::Marker::SPHERE;
//         marker.action = visualization_msgs::msg::Marker::ADD;

//         marker.pose.position.x = detection.x;
//         marker.pose.position.y = detection.y;

//         tf2::Quaternion q;
//         q.setRPY(0, 0, detection.yaw);
//         marker.pose.orientation = tf2::toMsg(q);

//         marker.scale.x = 0.05;
//         marker.scale.y = 0.05;
//         marker.scale.z = 1e-5;

//         marker.color.r = 0.0;
//         marker.color.g = 1.0;
//         marker.color.b = 0.0;
//         marker.color.a = 1.0;

//         marker_array.markers.push_back(marker);
//     }

//     detection_array_pub_->publish(detect_array_);
//     detected_opp_traj_marker_pub_->publish(marker_array);
// }

// int main(int argc, char **argv)
// {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<CollectDetection>();
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }
