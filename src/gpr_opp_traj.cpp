#include "pred_opp_traj/gpr_opp_traj.h"

GPROppTrajNode::GPROppTrajNode(std::string map, std::string pkg_path, int horizon, double dt)
{
    // this->declare_parameter("map", "");
    // std::string map = this->get_parameter("map").as_string();
    // std::string pkg_path = ament_index_cpp::get_package_share_directory("pred_opp_traj");
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
            if (first_line)
            {
                first_line = false;
                continue; // skip header
            }
            std::string value;
            std::stringstream ss(line);
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

    // this->declare_parameter("horizon", 0);
    // this->declare_parameter("dt", 0.0);
    // horizon_ = this->get_parameter("horizon").as_int();
    // dt_ = this->get_parameter("dt").as_double();
    horizon_ = horizon;
    dt_ = dt;

    // det_sub_ = this->create_subscription<pred_msgs::msg::Detection>("/detection", rclcpp::QoS(rclcpp::KeepLast(1)), std::bind(&GPROppTrajNode::detection_callback, this, std::placeholders::_1));
    // det_arr_sub_ = this->create_subscription<pred_msgs::msg::DetectionArray>("/detected_opp_traj", rclcpp::QoS(rclcpp::KeepLast(1)), std::bind(&GPROppTrajNode::detection_array_callback, this, std::placeholders::_1));

    // pred_pub_ = this->create_publisher<pred_msgs::msg::DetectionArray>("/pred_opp_traj", rclcpp::QoS(rclcpp::KeepLast(1)));
    // marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/pred_opp_traj_marker", rclcpp::QoS(rclcpp::KeepLast(1)));

    // timer_ = this->create_wall_timer(std::chrono::milliseconds(25), std::bind(&GPROppTrajNode::timer_callback, this));
}

GPPrediction perform_gp_regression(
    const std::vector<double>& t_samples,
    const std::vector<double>& y_samples,
    const std::vector<double>& y_var_samples,
    const std::vector<double>& pred_times,
    int kernel_type)
{
    // --- 1. Manual Y Normalization (equivalent to normalize_y=True) ---
    double y_mean = std::accumulate(y_samples.begin(), y_samples.end(), 0.0) / y_samples.size();
    std::vector<double> y_normalized = y_samples;
    for (auto& y : y_normalized)
        y -= y_mean;

    // --- 2. Initialize GP with kernel ---
    const int input_dim = 1;
    libgp::GaussianProcess* gp = nullptr;
    if (kernel_type == 0)
        gp = new libgp::GaussianProcess(input_dim, "CovMatern3iso");
    else
        gp = new libgp::GaussianProcess(input_dim, "CovSEiso");

    // --- 3. Set Hyperparameters ---
    int param_dim = gp->covf().get_param_dim();
    Eigen::VectorXd params(param_dim);
    double length_scale = 1.0;
    double signal_std   = std::sqrt(0.5);
    // Covariance functions in libgp expect log-hyperparameters
    params << std::log(length_scale), std::log(signal_std);
    gp->covf().set_loghyper(params);

    // --- 4. Add training samples ---
    for (size_t i = 0; i < t_samples.size(); ++i)
    {
        double x_arr[1] = {t_samples[i]};
        gp->add_pattern(x_arr, y_normalized[i]);
    }
    Eigen::Map<const Eigen::VectorXd> y_vars_vec(y_var_samples.data(), y_var_samples.size());
    gp->set_y_vars(y_vars_vec);

    // --- 5. Predictions ---
    GPPrediction result;
    for (double t : pred_times)
    {
        double x_arr[1] = {t};
        double pred_mean_norm = gp->f(x_arr);
        double pred_var = gp->var(x_arr);

        result.mean.push_back(pred_mean_norm + y_mean); // add normalization offset back
        result.std_dev.push_back(std::sqrt(std::max(pred_var, 1e-9))); // avoid sqrt of negative
    }

    delete gp;
    return result;
}

pred_msgs::msg::DetectionArray GPROppTrajNode::predict_trajectory(const pred_msgs::msg::Detection curr_opp, const pred_msgs::msg::DetectionArray det_arr)
{
    pred_msgs::msg::DetectionArray pred_opp_traj;

    if (curr_opp.v < 1.0)
    {
        for (int i = 0; i < horizon_; i++)
        {
          pred_msgs::msg::Detection d;
          d.dt = i * dt_;
          d.x = curr_opp.x;
          d.y = curr_opp.y;
          d.yaw = curr_opp.yaw;
          d.v = 0.0;
          d.x_var = 0.05;
          d.y_var = 0.05;
          d.yaw_var = 0.05;
          d.v_var = 0.05;
          pred_opp_traj.detections.push_back(d);
        }
    }
    else
    {
        double curr_opp_s = sp_.find_s(curr_opp.x, curr_opp.y, 0.0);
        std::cout << "curr_opp_s: " << curr_opp_s << ", x: " << curr_opp.x << ", y: " << curr_opp.y << std::endl;
        // int back_opp_idx = (int)(std::floor(curr_opp_s * 10)) % (int)det_arr.detections.size();
        int front_opp_idx = (int)(std::ceil(curr_opp_s * 10)) % (int)det_arr.detections.size();

        pred_msgs::msg::DetectionArray d_copy, sorted_d_array;
        d_copy = det_arr;
        for (int i = 0; i < (int)((horizon_ + 5) * dt_ * 10) * 15; i++)
        {
            int idx = (front_opp_idx + i) % (int)d_copy.detections.size();
            if (idx == front_opp_idx)
            {
                if (front_opp_idx - curr_opp_s * 10 < 0.0)
                    d_copy.detections[idx].dt = (front_opp_idx + (int)det_arr.detections.size() - curr_opp_s * 10) * d_copy.detections[idx].dt;
                else
                    d_copy.detections[idx].dt = (front_opp_idx - curr_opp_s * 10) * d_copy.detections[idx].dt;
            }
            else
            {
                if (idx == 0)
                    d_copy.detections[idx].dt = d_copy.detections[d_copy.detections.size() - 1].dt + d_copy.detections[idx].dt;
                else
                    d_copy.detections[idx].dt = d_copy.detections[idx - 1].dt + d_copy.detections[idx].dt;
            }
            sorted_d_array.detections.push_back(d_copy.detections[idx]);
        }

        std::vector<double> sorted_t, sorted_x, sorted_y, sorted_yaw, sorted_v, sorted_x_var, sorted_y_var, sorted_yaw_var, sorted_v_var;
        sorted_t.push_back(0.0);
        sorted_x.push_back(curr_opp.x);
        sorted_y.push_back(curr_opp.y);
        sorted_yaw.push_back(curr_opp.yaw);
        sorted_v.push_back(curr_opp.v);
        sorted_x_var.push_back(curr_opp.x_var);
        sorted_y_var.push_back(curr_opp.y_var);
        sorted_yaw_var.push_back(curr_opp.yaw_var);
        sorted_v_var.push_back(curr_opp.v_var);
        for (size_t i = 0; i < sorted_d_array.detections.size(); i++)
        {
            sorted_t.push_back(sorted_d_array.detections[i].dt);
            sorted_x.push_back(sorted_d_array.detections[i].x);
            sorted_y.push_back(sorted_d_array.detections[i].y);
            sorted_yaw.push_back(sorted_d_array.detections[i].yaw);
            sorted_v.push_back(sorted_d_array.detections[i].v);
            sorted_x_var.push_back(sorted_d_array.detections[i].x_var);
            sorted_y_var.push_back(sorted_d_array.detections[i].y_var);
            sorted_yaw_var.push_back(sorted_d_array.detections[i].yaw_var);
            sorted_v_var.push_back(sorted_d_array.detections[i].v_var);
        }

        // const double curr_v = 15.0; // Example value
        std::vector<double> pred_time;
        for (double t = 0.0; t < horizon_ * dt_; t += dt_) {
            // pred_time.push_back(t + curr_v * 0.01);
            pred_time.push_back(t);
        }

        GPPrediction pred_x = perform_gp_regression(sorted_t, sorted_x, sorted_x_var, pred_time, 0);
        GPPrediction pred_y = perform_gp_regression(sorted_t, sorted_y, sorted_y_var, pred_time, 0);
        GPPrediction pred_yaw = perform_gp_regression(sorted_t, sorted_yaw, sorted_yaw_var, pred_time, 0);
        GPPrediction pred_v = perform_gp_regression(sorted_t, sorted_v, sorted_v_var, pred_time, 1);

        for (int i = 0; i < horizon_; i++) {
            pred_msgs::msg::Detection d;
            d.dt   = i * dt_;
            d.x    = pred_x.mean[i];
            d.y    = pred_y.mean[i];
            d.yaw  = pred_yaw.mean[i];
            d.v    = pred_v.mean[i];
            d.x_var   = pred_x.std_dev[i];
            d.y_var   = pred_y.std_dev[i];
            d.yaw_var = pred_yaw.std_dev[i];
            d.v_var   = pred_v.std_dev[i];
            pred_opp_traj.detections.push_back(d);
        }
        // std::cout << "GPR Opponent Trajectory Prediction Time: " << (this->get_clock()->now().seconds() - curr_time) << std::endl;
    }
    return pred_opp_traj;
}

// void GPROppTrajNode::detection_callback(const pred_msgs::msg::Detection::SharedPtr msg)
// {
//     curr_opp_ = *msg;
//     is_curr_opp_ = true;
// }

// void GPROppTrajNode::detection_array_callback(const pred_msgs::msg::DetectionArray::SharedPtr msg)
// {
//     det_arr_ = *msg;
//     is_det_arr_ = true;
// }


// void GPROppTrajNode::timer_callback()
// {
//     if (!is_curr_opp_ || !is_det_arr_)
//         return;

//     double curr_time = this->get_clock()->now().seconds();
//     if (curr_opp_.v < 1.0)
//     {
//         pred_msgs::msg::DetectionArray pred_opp_traj;
//         visualization_msgs::msg::MarkerArray markers;
//         for (int i = 0; i < horizon_; i++)
//         {
//           pred_msgs::msg::Detection d;
//           d.dt = i * dt_;
//           d.x = curr_opp_.x;
//           d.y = curr_opp_.y;
//           d.yaw = curr_opp_.yaw;
//           d.v = 0.0;
//           d.x_var = 0.05;
//           d.y_var = 0.05;
//           d.yaw_var = 0.05;
//           d.v_var = 0.05;
//           pred_opp_traj.detections.push_back(d);

//           visualization_msgs::msg::Marker m;
//           m.header.frame_id = "map";
//           m.header.stamp = this->get_clock()->now();
//           m.ns = "pred_opp_traj";
//           m.id = i;
//           m.type = visualization_msgs::msg::Marker::SPHERE;
//           m.action = visualization_msgs::msg::Marker::ADD;
//           m.pose.position.x = d.x;
//           m.pose.position.y = d.y;
//           m.pose.orientation.w = 1.0;
//           m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 1e-5;
//           m.color.r = curr_opp_.v / 10.0;
//           m.color.g = 0.0;
//           m.color.b = -(curr_opp_.v - 10.0) / 10.0;
//           m.color.a = 1.0;
//           markers.markers.push_back(m);

//           m.id = i + 100;
//           m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 1e-5;
//           m.color.r = 0.0;
//           m.color.g = 0.0;
//           m.color.b = 0.0;
//           m.color.a = 0.1;
//           markers.markers.push_back(m);
//         }
//         pred_pub_->publish(pred_opp_traj);
//         marker_pub_->publish(markers);
//         // std::cout << "GPR Opponent Trajectory Prediction Time: " << (this->get_clock()->now().seconds() - curr_time) << std::endl;
//     }
//     else
//     {
//         double curr_opp_s = sp_.find_s(curr_opp_.x, curr_opp_.y, 0.0);
//         // int back_opp_idx = (int)(std::floor(curr_opp_s * 10)) % (int)det_arr_.detections.size();
//         int front_opp_idx = (int)(std::ceil(curr_opp_s * 10)) % (int)det_arr_.detections.size();

//         pred_msgs::msg::DetectionArray d_copy, sorted_d_array;
//         d_copy = det_arr_;
//         for (int i = 0; i < (int)((horizon_ + 5) * dt_ * 10) * 9; i++)
//         {
//             int idx = (front_opp_idx + i) % (int)d_copy.detections.size();
//             if (idx == front_opp_idx)
//             {
//                 if (front_opp_idx - curr_opp_s * 10 < 0.0)
//                     d_copy.detections[idx].dt = (front_opp_idx + (int)det_arr_.detections.size() - curr_opp_s * 10) * d_copy.detections[idx].dt;
//                 else
//                     d_copy.detections[idx].dt = (front_opp_idx - curr_opp_s * 10) * d_copy.detections[idx].dt;
//             }
//             else
//             {
//                 if (idx == 0)
//                     d_copy.detections[idx].dt = d_copy.detections[d_copy.detections.size() - 1].dt + d_copy.detections[idx].dt;
//                 else
//                     d_copy.detections[idx].dt = d_copy.detections[idx - 1].dt + d_copy.detections[idx].dt;
//             }
//             sorted_d_array.detections.push_back(d_copy.detections[idx]);
//         }

//         std::vector<double> sorted_t, sorted_x, sorted_y, sorted_yaw, sorted_v, sorted_x_var, sorted_y_var, sorted_yaw_var, sorted_v_var;
//         sorted_t.push_back(0.0);
//         sorted_x.push_back(curr_opp_.x);
//         sorted_y.push_back(curr_opp_.y);
//         sorted_yaw.push_back(curr_opp_.yaw);
//         sorted_v.push_back(curr_opp_.v);
//         sorted_x_var.push_back(0.01);
//         sorted_y_var.push_back(0.01);
//         sorted_yaw_var.push_back(0.01);
//         sorted_v_var.push_back(0.01);
//         for (size_t i = 0; i < sorted_d_array.detections.size(); i++)
//         {
//             sorted_t.push_back(sorted_d_array.detections[i].dt);
//             sorted_x.push_back(sorted_d_array.detections[i].x);
//             sorted_y.push_back(sorted_d_array.detections[i].y);
//             sorted_yaw.push_back(sorted_d_array.detections[i].yaw);
//             sorted_v.push_back(sorted_d_array.detections[i].v);
//             sorted_x_var.push_back(sorted_d_array.detections[i].x_var);
//             sorted_y_var.push_back(sorted_d_array.detections[i].y_var);
//             sorted_yaw_var.push_back(sorted_d_array.detections[i].yaw_var);
//             sorted_v_var.push_back(sorted_d_array.detections[i].v_var);
//         }

//         // const double curr_v = 15.0; // Example value
//         std::vector<double> pred_time;
//         for (double t = 0.0; t < horizon_ * dt_; t += dt_) {
//             // pred_time.push_back(t + curr_v * 0.01);
//             pred_time.push_back(t);
//         }

//         GPPrediction pred_x = perform_gp_regression(sorted_t, sorted_x, sorted_x_var, pred_time, 0);
//         GPPrediction pred_y = perform_gp_regression(sorted_t, sorted_y, sorted_y_var, pred_time, 0);
//         GPPrediction pred_yaw = perform_gp_regression(sorted_t, sorted_yaw, sorted_yaw_var, pred_time, 0);
//         GPPrediction pred_v = perform_gp_regression(sorted_t, sorted_v, sorted_v_var, pred_time, 1);

//         pred_msgs::msg::DetectionArray pred_opp_traj;
//         visualization_msgs::msg::MarkerArray markers;
//         for (int i = 0; i < horizon_; i++) {
//             pred_msgs::msg::Detection d;
//             d.dt   = i * dt_;
//             d.x    = pred_x.mean[i];
//             d.y    = pred_y.mean[i];
//             d.yaw  = pred_yaw.mean[i];
//             d.v    = pred_v.mean[i];
//             d.x_var   = pred_x.std_dev[i];
//             d.y_var   = pred_y.std_dev[i];
//             d.yaw_var = pred_yaw.std_dev[i];
//             d.v_var   = pred_v.std_dev[i];
//             pred_opp_traj.detections.push_back(d);

//             visualization_msgs::msg::Marker m;
//             m.header.frame_id = "map";
//             m.header.stamp = this->get_clock()->now();
//             m.ns = "pred_opp_traj";
//             m.id = i;
//             m.type = visualization_msgs::msg::Marker::SPHERE;
//             m.action = visualization_msgs::msg::Marker::ADD;
//             m.pose.position.x = d.x;
//             m.pose.position.y = d.y;
//             m.pose.orientation.w = 1.0;
//             m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.01;
//             m.color.r = curr_opp_.v / 10.0;
//             m.color.g = 0.0;
//             m.color.b = -(curr_opp_.v - 10.0) / 10.0;
//             m.color.a = 1.0;
//             markers.markers.push_back(m);

//             m.id = i + 100;
//             m.scale.x = d.x_var * 5;
//             m.scale.y = d.y_var * 5;
//             m.scale.z = 0.0;
//             m.color.r = 0.0;
//             m.color.g = 0.0;
//             m.color.b = 0.0;
//             m.color.a = 0.1;
//             markers.markers.push_back(m);
//         }

//         pred_pub_->publish(pred_opp_traj);
//         marker_pub_->publish(markers);
//         // std::cout << "GPR Opponent Trajectory Prediction Time: " << (this->get_clock()->now().seconds() - curr_time) << std::endl;
//     }
// }

// int main(int argc, char **argv) {
//   rclcpp::init(argc, argv);
//   rclcpp::spin(std::make_shared<GPROppTrajNode>());
//   rclcpp::shutdown();
//   return 0;
// }