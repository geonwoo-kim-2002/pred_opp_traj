from math import *
import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from pred_msgs.msg import Detection

def detection(scan: LaserScan, ego_odom: Odometry, opp_odom: Odometry) -> Detection:
    ego_x = ego_odom.pose.pose.position.x
    ego_y = ego_odom.pose.pose.position.y
    ego_quat = ego_odom.pose.pose.orientation
    opp_x = opp_odom.pose.pose.position.x
    opp_y = opp_odom.pose.pose.position.y
    opp_quat = opp_odom.pose.pose.orientation
    closest = 100
    closest_x = 0.0
    closest_y = 0.0
    yaw = R.from_quat([ego_quat.x, ego_quat.y, ego_quat.z, ego_quat.w]).as_euler('zyx', degrees=False)[0]
    for i in range(len(scan.ranges)):
        angle = scan.angle_min + i * scan.angle_increment

        scan_range = scan.ranges[i]
        scan_x = ego_x + scan_range * cos(yaw + angle)
        scan_y = ego_y + scan_range * sin(yaw + angle)

        distance = hypot(scan_x - opp_x, scan_y - opp_y)
        if distance < closest:
            closest = distance
            closest_x = scan_x
            closest_y = scan_y

        if closest < 0.3:
            detection_msg = Detection()
            detection_msg.dt = 0.
            detection_msg.x = opp_x
            detection_msg.y = opp_y
            detection_msg.yaw = R.from_quat([opp_quat.x, opp_quat.y, opp_quat.z, opp_quat.w]).as_euler('zyx', degrees=False)[0]
            detection_msg.v = hypot(opp_odom.twist.twist.linear.x, opp_odom.twist.twist.linear.y)

            detection_msg.x_var = 0.05
            detection_msg.y_var = 0.05
            detection_msg.yaw_var = 0.05
            detection_msg.v_var = 0.05
            return detection_msg
    print(closest, closest_x, closest_y, flush=True)
    print(f'opp_x: {opp_x}, opp_y: {opp_y}, ego_x: {ego_x}, ego_y: {ego_y}', flush=True)
    return None