#ifndef TRACK_H
#define TRACK_H

#include <fstream>

#include "LocalPath/CubicSpline2D.h"
#include "f1_msgs/msg/waypoint_array.hpp"

using namespace std;

struct lane_info {
    double s;
    double left_width;
    double right_width;
};

class Track
{
public:
    Track(){};
    Track(string centerline_filename, string width_filename);
    ~Track();
    CubicSpline2D csp;
    f1_msgs::msg::WaypointArray global_path;
    vector<lane_info> lane;

    void unwrapSplineLength(double s_max, double &spline_length);
private:
    void readCenterline(const string &filename);
    void readWidthInfo(const string &filename);
    void setCubicSpline();
};

#endif  // TRACK_H