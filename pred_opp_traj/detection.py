import rclpy
from rclpy.node import Node

from math import *
import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from pred_msgs.msg import Detection
# std_msgs/Header header
# float32 x
# float32 y
# float32 yaw
# float32 v

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.laser_subscriber = self.create_subscription(LaserScan,'/scan',self.laser_callback,10)
        self.ego_odom_subscriber = self.create_subscription(Odometry,'/ego_racecar/odom',self.ego_odom_callback,10)
        self.opp_odom_subscriber = self.create_subscription(Odometry,'/ego_racecar/opp_odom',self.opp_odom_callback,10)

        self.detect_pub = self.create_publisher(Detection, '/detection', 10)
        self.detect_marker_pub = self.create_publisher(Marker, '/detection_marker', 10)

        self.is_scan = False
        self.is_ego_odom = False
        self.is_opp_odom = False
        self.timer = self.create_timer(0.001, self.timer_callback)

    def laser_callback(self, msg):
        self.is_scan = True
        self.scan = msg

    def ego_odom_callback(self, msg):
        self.is_ego_odom = True
        self.ego_odom = msg

    def opp_odom_callback(self, msg):
        self.is_opp_odom = True
        self.opp_odom = msg

    def timer_callback(self):
        if self.is_scan and self.is_ego_odom and self.is_opp_odom:
            ego_x = self.ego_odom.pose.pose.position.x
            ego_y = self.ego_odom.pose.pose.position.y
            ego_quat = self.ego_odom.pose.pose.orientation
            opp_x = self.opp_odom.pose.pose.position.x
            opp_y = self.opp_odom.pose.pose.position.y
            opp_quat = self.opp_odom.pose.pose.orientation

            closest = 100
            for i in range(len(self.scan.ranges)):
                angle = self.scan.angle_min + i * self.scan.angle_increment
                yaw = R.from_quat([ego_quat.x, ego_quat.y, ego_quat.z, ego_quat.w]).as_euler('zyx', degrees=False)[0]

                scan_range = self.scan.ranges[i]
                scan_x = ego_x + scan_range * cos(yaw + angle)
                scan_y = ego_y + scan_range * sin(yaw + angle)

                distance = hypot(scan_x - opp_x, scan_y - opp_y)
                if distance < closest:
                    closest = distance

                if closest < 0.3:
                    detection_msg = Detection()
                    detection_msg.header.stamp = self.get_clock().now().to_msg()
                    detection_msg.x = opp_x
                    detection_msg.y = opp_y
                    # print("angle: ", R.from_quat([opp_quat.w, opp_quat.x, opp_quat.y, opp_quat.z]).as_euler('zyx', degrees=False))
                    # print("scan_x: ", scan_x, "scan_y: ", scan_y, "range: ", scan_range)
                    # print("opp_x: ", opp_x, "opp_y: ", opp_y)
                    detection_msg.yaw = R.from_quat([opp_quat.x, opp_quat.y, opp_quat.z, opp_quat.w]).as_euler('zyx', degrees=False)[0]
                    detection_msg.v = hypot(self.opp_odom.twist.twist.linear.x, self.opp_odom.twist.twist.linear.y)
                    self.detect_pub.publish(detection_msg)

                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.id = 0
                    marker.type = Marker.ARROW
                    marker.action = Marker.ADD
                    marker.pose.position.x = opp_x
                    marker.pose.position.y = opp_y
                    marker.pose.orientation = opp_quat
                    marker.scale.x = max(0.5, hypot(self.opp_odom.twist.twist.linear.x, self.opp_odom.twist.twist.linear.y) * 0.5)
                    marker.scale.y = 0.2
                    marker.scale.z = 0.2
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    self.detect_marker_pub.publish(marker)

                    break

            print("closest: ", hypot(ego_x - opp_x, ego_y - opp_y))

def main():
    rclpy.init()
    detection_node = DetectionNode()
    rclpy.spin(detection_node)
    detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()