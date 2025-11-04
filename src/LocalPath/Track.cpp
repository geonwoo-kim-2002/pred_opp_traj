#include "LocalPath/Track.h"

Track::Track(string centerline_filename, string width_filename)
{
    readCenterline(centerline_filename);
    readWidthInfo(width_filename);
    setCubicSpline();
}

Track::~Track() {}

void Track::readCenterline(const string &filename)
{
    ifstream read_file;
    cout << "center filename : " << filename << endl;
    read_file.open(filename);
    if (read_file.is_open())
    {
        bool first_line = true;
        while (!read_file.eof())
        {
            string line;
            getline(read_file, line);
            if (first_line)
            {
                first_line = false;
                continue; // skip header
            }
            if (line.empty())
                continue;

            string buffer;
            istringstream iss(line);
            vector<string> res;
            while (getline(iss, buffer, ','))
            {
                res.push_back(buffer);
            }

            f1_msgs::msg::Waypoint pos;
            pos.x = stof(res[0]);
            pos.y = stof(res[1]);
            pos.v = 0;
            pos.vm = 0;

            global_path.wp.push_back(pos);
        }
    }
    cout << "global path size: " << global_path.wp.size() << endl;
    read_file.close();
}

void Track::readWidthInfo(const string &filename)
{
    ifstream read_file;
    cout << "width filename : " << filename << endl;
    read_file.open(filename);
    if (read_file.is_open())
    {
        bool first_line = true;
        while (!read_file.eof())
        {
            string line;

            getline(read_file, line);
            if (first_line)
            {
                first_line = false;
                continue; // skip header
            }
            if (line.empty())
                continue;

            string buffer;
            istringstream iss(line);
            vector<string> res;
            while (getline(iss, buffer, ','))
            {
                res.push_back(buffer);
            }

            lane_info info;
            info.s = stof(res[0]);
            info.left_width = stof(res[1]);
            info.right_width = stof(res[2]);
            lane.push_back(info);
        }
    }
    cout << "lane info size: " << lane.size() << endl;
    read_file.close();
}

void Track::setCubicSpline()
{
    vector<f1_msgs::msg::Waypoint> sample_path;
    for (std::size_t idx = 0; idx < global_path.wp.size(); idx++)
    {
        sample_path.push_back(global_path.wp[idx]);
    }
    csp.setCubicSpline2D(sample_path);
}

void Track::unwrapSplineLength(double s_max, double &spline_length)
{
    if (spline_length < -s_max / 2)
        spline_length += s_max;
    else if (spline_length > s_max / 2)
        spline_length -= s_max;
}