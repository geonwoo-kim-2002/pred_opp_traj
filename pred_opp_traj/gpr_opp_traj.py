import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import pandas as pd
from math import *
import copy
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

from pred_opp_traj.Spline import Spline, Spline2D
from pred_msgs.msg import Detection, DetectionArray

class GPROppTraj(Node):
    def __init__(self):
        super().__init__('gpr_opp_traj_node')
        self.declare_parameter('map', '')
        self.map = self.get_parameter('map').value
        self.pkg_path = get_package_share_directory('pred_opp_traj')
        center_path = pd.read_csv(f'{self.pkg_path}/data/path/{self.map}_path.csv')
        self.sp = Spline2D(center_path['x_m'], center_path['y_m'])

        self.declare_parameter('horizon', 0)
        self.horizon = self.get_parameter('horizon').value
        self.declare_parameter('dt', 0.0)
        self.dt = self.get_parameter('dt').value

        self.curr_opp = None
        self.prev_opp = None
        self.detection_array = None
        self.detection_sub = self.create_subscription(Detection, '/detection', self.detection_callback, 1)
        self.detection_array_sub = self.create_subscription(DetectionArray, '/detected_opp_traj', self.detection_array_callback, 1)

        self.pred_opp_traj = None
        self.pred_opp_traj_pub = self.create_publisher(DetectionArray, '/pred_opp_traj', 1)
        self.pred_opp_traj_marker_pub = self.create_publisher(MarkerArray, '/pred_opp_traj_marker', 1)

        self.timer = self.create_timer(0.001, self.timer_callback)

    def detection_callback(self, msg):
        self.curr_opp = msg
        # print("detection callback")

    def detection_array_callback(self, msg):
        self.detection_array = msg
        # print("detection array callback")

    def timer_callback(self):
        curr_time = time.time()
        if self.curr_opp is not None and self.detection_array is not None:
            if self.curr_opp.v < 0.5:
                self.pred_opp_traj = DetectionArray()
                marker_array = MarkerArray()
                for i in range(self.horizon):
                    detection = Detection()
                    detection.dt = i * self.dt
                    detection.x = self.curr_opp.x
                    detection.y = self.curr_opp.y
                    detection.yaw = self.curr_opp.yaw
                    detection.v = self.curr_opp.v
                    detection.x_var = 0.05
                    detection.y_var = 0.05
                    detection.yaw_var = 0.05
                    detection.v_var = 0.05
                    self.pred_opp_traj.detections.append(detection)

                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = 'pred_opp_traj'
                    marker.id = i
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    marker.pose.position.x = self.curr_opp.x
                    marker.pose.position.y = self.curr_opp.y
                    marker.pose.orientation.w = 1.0

                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 1e-5

                    marker.color.r = self.curr_opp.v / 10.0
                    marker.color.g = 0.0
                    marker.color.b = -(self.curr_opp.v - 10.0) / 10.0
                    marker.color.a = 1.0
                    marker_array.markers.append(marker)

                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = 'pred_opp_traj'
                    marker.id = i + 100
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    marker.pose.position.x = self.curr_opp.x
                    marker.pose.position.y = self.curr_opp.y
                    marker.pose.orientation.w = 1.0

                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.0

                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.1
                    marker_array.markers.append(marker)

                self.pred_opp_traj_pub.publish(self.pred_opp_traj)
                self.pred_opp_traj_marker_pub.publish(marker_array)
                self.prev_opp = self.curr_opp
                print("GPR Opponent Trajectory Prediction Time:", time.time() - curr_time)

            else:
                curr_opp_s = self.sp.find_s(self.curr_opp.x, self.curr_opp.y)
                back_opp_idx = int(floor(curr_opp_s * 10)) % len(self.detection_array.detections)
                front_opp_idx = int(ceil(curr_opp_s * 10)) % len(self.detection_array.detections)
                # print("Current Opponent S:", curr_opp_s, "Back Index:", back_opp_idx, "Front Index:", front_opp_idx)

                da_copy = copy.deepcopy(self.detection_array.detections)
                sorted_detection_array = DetectionArray()
                for i in range(int((self.horizon + 5) * self.dt * 10) * 10):
                    idx = (front_opp_idx + i) % len(da_copy)
                    if idx == front_opp_idx:
                        da_copy[idx].dt = (1 - (curr_opp_s * 10 - back_opp_idx)) * da_copy[idx].dt
                    else:
                        da_copy[idx].dt = da_copy[idx - 1].dt + da_copy[idx].dt
                    sorted_detection_array.detections.append(da_copy[idx])

                x_var = np.hstack((np.array([0.0]), np.array([d.x_var for d in sorted_detection_array.detections])))
                y_var = np.hstack((np.array([0.0]), np.array([d.y_var for d in sorted_detection_array.detections])))
                yaw_var = np.hstack((np.array([0.0]), np.array([d.yaw_var for d in sorted_detection_array.detections])))
                v_var = np.hstack((np.array([0.0]), np.array([d.v_var for d in sorted_detection_array.detections])))
                c_kernel_x = ConstantKernel(0.5, constant_value_bounds=(1e-3, 1e3))
                c_kernel_y = ConstantKernel(0.5, constant_value_bounds=(1e-3, 1e3))
                c_kernel_yaw = ConstantKernel(0.5, constant_value_bounds=(1e-3, 1e3))
                c_kernel_v = ConstantKernel(0.5, constant_value_bounds=(1e-3, 1e3))
                kernel_x = c_kernel_x * Matern(length_scale=1.0, nu=3/2)
                kernel_y = c_kernel_y * Matern(length_scale=1.0, nu=3/2)
                kernel_yaw = c_kernel_yaw * Matern(length_scale=1.0, nu=3/2)
                kernel_v = c_kernel_v * RBF(length_scale=1.0)

                gp_x = GaussianProcessRegressor(kernel=kernel_x, alpha=x_var, normalize_y=True, optimizer=None)
                gp_y = GaussianProcessRegressor(kernel=kernel_y, alpha=y_var, normalize_y=True, optimizer=None)
                gp_yaw = GaussianProcessRegressor(kernel=kernel_yaw, alpha=yaw_var, normalize_y=True, optimizer=None)
                gp_v = GaussianProcessRegressor(kernel=kernel_v, alpha=v_var, normalize_y=True, optimizer=None)

                gp_x.fit(np.hstack((np.array([0.0]), np.array([d.dt for d in sorted_detection_array.detections]))).reshape(-1, 1), np.hstack((np.array([self.curr_opp.x]), np.array([d.x for d in sorted_detection_array.detections]))))
                gp_y.fit(np.hstack((np.array([0.0]), np.array([d.dt for d in sorted_detection_array.detections]))).reshape(-1, 1), np.hstack((np.array([self.curr_opp.y]), np.array([d.y for d in sorted_detection_array.detections]))))
                gp_yaw.fit(np.hstack((np.array([0.0]), np.array([d.dt for d in sorted_detection_array.detections]))).reshape(-1, 1), np.hstack((np.array([self.curr_opp.yaw]), np.array([d.yaw for d in sorted_detection_array.detections]))))
                gp_v.fit(np.hstack((np.array([0.0]), np.array([d.dt for d in sorted_detection_array.detections]))).reshape(-1, 1), np.hstack((np.array([self.curr_opp.v]), np.array([d.v for d in sorted_detection_array.detections]))))

                pred_time = np.arange(0, self.horizon * self.dt, self.dt) + self.curr_opp.v * 0.01
                pred_x, x_std = gp_x.predict(pred_time.reshape(-1, 1), return_std=True)
                pred_y, y_std = gp_y.predict(pred_time.reshape(-1, 1), return_std=True)
                pred_yaw, yaw_std = gp_yaw.predict(pred_time.reshape(-1, 1), return_std=True)
                pred_v, v_std = gp_v.predict(pred_time.reshape(-1, 1), return_std=True)

                self.pred_opp_traj = DetectionArray()
                marker_array = MarkerArray()
                for i in range(len(pred_x)):
                    detection = Detection()
                    detection.dt = pred_time[i]
                    detection.x = pred_x[i]
                    detection.y = pred_y[i]
                    detection.yaw = pred_yaw[i]
                    detection.v = pred_v[i]
                    detection.x_var = x_std[i]
                    detection.y_var = y_std[i]
                    detection.yaw_var = yaw_std[i]
                    detection.v_var = v_std[i]
                    self.pred_opp_traj.detections.append(detection)

                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = 'pred_opp_traj'
                    marker.id = i
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    marker.pose.position.x = pred_x[i]
                    marker.pose.position.y = pred_y[i]
                    # quat = R.from_euler('z', detection.yaw).as_quat()
                    # marker.pose.orientation.x = quat[0]
                    # marker.pose.orientation.y = quat[1]
                    # marker.pose.orientation.z = quat[2]
                    # marker.pose.orientation.w = quat[3]
                    marker.pose.orientation.w = 1.0

                    # marker.scale.x = max(0.1, detection.v * 0.01)
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 1e-5

                    # marker.color.r = max((pred_v[i] - 5.0) / 5.0, 0.0)
                    # marker.color.g = (-abs(pred_v[i] - 5.0) + 5.0) / 5.0
                    # marker.color.b = max(-(pred_v[i] - 5.0) / 5.0, 0.0)
                    marker.color.r = pred_v[i] / 10.0
                    marker.color.g = 0.0
                    marker.color.b = -(pred_v[i] - 10.0) / 10.0
                    marker.color.a = 1.0
                    marker_array.markers.append(marker)

                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = 'pred_opp_traj'
                    marker.id = i + 100
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    marker.pose.position.x = pred_x[i]
                    marker.pose.position.y = pred_y[i]
                    marker.pose.orientation.w = 1.0

                    marker.scale.x = x_std[i] * 2
                    marker.scale.y = y_std[i] * 2
                    marker.scale.z = 0.0

                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.1
                    marker_array.markers.append(marker)
                self.pred_opp_traj_pub.publish(self.pred_opp_traj)
                self.pred_opp_traj_marker_pub.publish(marker_array)
                self.prev_opp = self.curr_opp
                print("GPR Opponent Trajectory Prediction Time:", time.time() - curr_time, len(pred_x))

def main():
    rclpy.init()
    gpr_opp_traj_node = GPROppTraj()
    rclpy.spin(gpr_opp_traj_node)
    gpr_opp_traj_node.destroy_node()
    rclpy.shutdown()