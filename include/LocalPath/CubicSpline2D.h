#ifndef FRENET_OPTIMAL_TRAJECTORY_CUBICSPLINE2D_H
#define FRENET_OPTIMAL_TRAJECTORY_CUBICSPLINE2D_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "LocalPath/CubicSpline1D.h"

#include "f1_msgs/msg/waypoint.hpp"

using namespace std;

// 2-dimensional cubic spline class.
// For technical details see: http://mathworld.wolfram.com/CubicSpline.html
class CubicSpline2D {
  public:
    CubicSpline2D();
    CubicSpline2D(const vector<f1_msgs::msg::Waypoint> &global_path);
    void setCubicSpline2D(const vector<f1_msgs::msg::Waypoint> &global_path);

    void
    visualize_filtered_points(const vector<vector<double>> filtered_points);
    double calc_x(double t);
    double calc_xdot(double t);
    double calc_y(double t);
    double calc_ydot(double t);
    double calc_curvature(double t);
    double calc_yaw(double t);
    double find_s(double x, double y,
                  double s0); // find cloest position(s) of one global
                              // position(x,y) in cubic spine
    double local_find_s(double x, double y, double s0, double search_range);
    double calc_lateral_deviation(double x, double y, double s0);

    vector<vector<double>> filtered_points;

    vector<double> s;
    CubicSpline1D sx, sy;

  private:
    void calc_s(
        const vector<double> &x,
        const vector<double> &y); // find position s values for all given x, y
    vector<vector<double>>
    remove_collinear_points(const vector<f1_msgs::msg::Waypoint> &global_path);
    bool are_collinear(double x1, double y1, double x2, double y2, double x3, double y3);
};

#endif // FRENET_OPTIMAL_TRAJECTORY_CUBICSPLINE2D_H
