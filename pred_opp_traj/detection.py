import rclpy
from rclpy.node import Node

from math import *
import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import PoseStamped

from pred_msgs.msg import Detection
# float32 dt

# float32 x
# float32 y
# float32 yaw
# float32 v

# float32 x_var
# float32 y_var
# float32 yaw_var
# float32 v_var

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.declare_parameter("is_simulation", True)
        self.is_simulation = self.get_parameter("is_simulation").value

        self.is_scan = False
        self.is_ego_odom = False
        self.is_opp_odom = False
        self.opp_boxes = Detection2DArray()
        
        if self.is_simulation:
            self.laser_subscriber = self.create_subscription(LaserScan,'/scan',self.laser_callback,1)
            self.ego_odom_subscriber = self.create_subscription(Odometry,'/ego_racecar/odom',self.ego_odom_callback,1)
            self.opp_odom_subscriber = self.create_subscription(Odometry,'/ego_racecar/opp_odom',self.opp_odom_callback,1)
        else:
            self.laser_subscriber = self.create_subscription(LaserScan,'/scan',self.laser_callback,1)
            self.ego_pose_subscriber = self.create_subscription(PoseStamped,'/mcl_pose',self.ego_pose_callback,1)
            self.opp_box_subscriber = self.create_subscription(Detection2DArray,'/bounding_box',self.opp_box_callback,1)            

        self.detect_pub = self.create_publisher(Detection, '/detection', 1)
        self.detect_marker_pub = self.create_publisher(Marker, '/detection_marker', 1)

        self.timer = self.create_timer(0.001, self.timer_callback)

    def laser_callback(self, msg):
        self.is_scan = True
        self.scan = msg
        # print("laser callback", flush=True)

    def ego_odom_callback(self, msg):
        self.is_ego_odom = True
        self.ego_odom = msg

    def opp_odom_callback(self, msg):
        self.is_opp_odom = True
        self.opp_odom = msg
    
    def ego_pose_callback(self, msg):
        self.is_ego_odom = True
        self.ego_pose = msg
        # print("ego callback", flush=True)
        
    def opp_box_callback(self, msg):
        self.opp_boxes = msg

    def timer_callback(self):
        if self.is_simulation:
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
                        detection_msg.dt = 0.
                        detection_msg.x = opp_x
                        detection_msg.y = opp_y
                        # print("angle: ", R.from_quat([opp_quat.w, opp_quat.x, opp_quat.y, opp_quat.z]).as_euler('zyx', degrees=False))
                        # print("scan_x: ", scan_x, "scan_y: ", scan_y, "range: ", scan_range)

                        detection_msg.yaw = R.from_quat([opp_quat.x, opp_quat.y, opp_quat.z, opp_quat.w]).as_euler('zyx', degrees=False)[0]
                        detection_msg.v = hypot(self.opp_odom.twist.twist.linear.x, self.opp_odom.twist.twist.linear.y)

                        detection_msg.x_var = 0.05
                        detection_msg.y_var = 0.05
                        detection_msg.yaw_var = 0.05
                        detection_msg.v_var = 0.05

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
                        marker.scale.x = hypot(self.opp_odom.twist.twist.linear.x, self.opp_odom.twist.twist.linear.y) * 0.2
                        marker.scale.y = 0.2
                        marker.scale.z = 0.2
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                        marker.color.a = 1.0
                        self.detect_marker_pub.publish(marker)

                        break
        else:
            if self.is_scan and self.is_ego_odom:
                if len(self.opp_boxes.detections) > 0:
                    ego_x = self.ego_pose.pose.position.x
                    ego_y = self.ego_pose.pose.position.y
                    ego_quat = self.ego_pose.pose.orientation
                    ego_yaw = R.from_quat([ego_quat.x, ego_quat.y, ego_quat.z, ego_quat.w]).as_euler('zyx', degrees=False)[0]
                    
                    opp_local_x = self.opp_boxes.detections[0].bbox.center.position.x
                    opp_local_y = self.opp_boxes.detections[0].bbox.center.position.y
                                       
                    opp_x = ego_x + opp_local_x * cos(ego_yaw) - opp_local_y * sin(ego_yaw)
                    opp_y = ego_y + opp_local_x * sin(ego_yaw) + opp_local_y * cos(ego_yaw)
                    
                    detection_msg = Detection()
                    detection_msg.dt = 0.
                    detection_msg.x = opp_x
                    detection_msg.y = opp_y
                    detection_msg.yaw = 0.0
                    detection_msg.v = 0.0
                    detection_msg.x_var = 0.05
                    detection_msg.y_var = 0.05
                    detection_msg.yaw_var = 0.05
                    detection_msg.v_var = 0.05
                    self.detect_pub.publish(detection_msg)

                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.id = 0
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose.position.x = opp_x
                    marker.pose.position.y = opp_y
                    # marker.pose.orientation = opp_quat
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.2
                    marker.scale.y = 0.2
                    marker.scale.z = 1e-5
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    self.detect_marker_pub.publish(marker)
                    # print("detect opp", flush=True)
                # else:
                    # print("no detection", flush=True)

def main():
    rclpy.init()
    detection_node = DetectionNode()
    rclpy.spin(detection_node)
    detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()