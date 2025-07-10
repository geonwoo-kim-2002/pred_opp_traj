import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import time

# from .Spline import Spline2D
from pred_opp_traj.Spline import Spline, Spline2D
from pred_msgs.msg import Detection

class CollectDetection(Node):
    def __init__(self):
        super().__init__('collect_detection_node')
        self.pkg_path = get_package_share_directory('pred_opp_traj')

        self.declare_parameter('map', 'levinelobby')
        self.done_init = False
        self.detections = []
        self.init_detections()

        self.prev_detection = None
        self.first_point = True
        self.prev_time = 0.0
        self.prev_opp_idx = None
        self.detection_sub = self.create_subscription(Detection, '/detection', self.detection_callback, 10)

        self.detected_opp_traj_pub = self.create_publisher(MarkerArray, '/detected_opp_traj', 10)
        self.timer = self.create_timer(0.001, self.timer_callback)

    def init_detections(self):
        self.map = self.get_parameter('map').value
        center_path = pd.read_csv(f'{self.pkg_path}/data/path/{self.map}_path.csv')
        self.sp = Spline2D(center_path['x_m'], center_path['y_m'])

        x_center = []
        y_center = []
        for i in np.arange(0.0, self.sp.s[-1], 0.1):
            ix, iy = self.sp.calc_position(i)
            x_center.append(ix)
            y_center.append(iy)

        states = pd.read_csv(f'{self.pkg_path}/data/raceline/{self.map}_states.csv')
        if len(x_center) != len(states['t_s']):
            raise ValueError(f"Length of center path does not match length of states data. spline length: {len(x_center)}, states length: {len(states['t_s'])}")

        raceline = pd.read_csv(f'{self.pkg_path}/data/raceline/{self.map}_traj_race_cl.csv')
        sp_race = Spline2D(raceline['x_m'], raceline['y_m'])
        sp_race_v = Spline(sp_race.s, raceline['vx_mps'])
        race_x = []
        race_y = []
        race_yaw = []
        race_v = []
        for i in np.arange(0.0, sp_race.s[-1], 0.01):
            ix, iy = sp_race.calc_position(i)
            iyaw = sp_race.calc_yaw(i)
            race_x.append(ix)
            race_y.append(iy)
            race_yaw.append(iyaw)
            race_v.append(sp_race_v.calc(i))

        for i, (cx, cy) in enumerate(zip(x_center, y_center)):
            min_dist = 1000.
            min_idx = -1
            for j in range(len(race_x)):
                dist = np.linalg.norm(np.array([cx, cy]) - np.array([race_x[j], race_y[j]]))
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j

            new_detection = Detection()
            if i == 0:
                new_detection.dt = states['t_s'][len(states['t_s']) - 1] - states['t_s'][len(states['t_s']) - 2]
            else:
                new_detection.dt = states['t_s'][i] - states['t_s'][i - 1]
            new_detection.x = race_x[min_idx]
            new_detection.y = race_y[min_idx]
            new_detection.yaw = race_yaw[min_idx]
            new_detection.v = race_v[min_idx]
            new_detection.x_var = 10.
            new_detection.y_var = 10.
            new_detection.yaw_var = 3.141592
            new_detection.v_var = 5.

            self.detections.append(new_detection)
        self.done_init = True
        print("Detections initialized.")

    def detection_callback(self, msg):
        if time.time() - self.prev_time >= 0.5:
            # print("time duration:", time.time() - self.prev_time)
            self.first_point = True

        if msg != self.prev_detection and self.done_init:
            curr_opp_s = self.sp.find_s(msg.x, msg.y)
            if (curr_opp_s * 100) % 10 <= 1 or (curr_opp_s * 100) % 10 >= 8:
                opp_idx = int(round(curr_opp_s, 1) * 10)
                # print("prev_opp_idx:", self.prev_opp_idx, "opp_idx:", opp_idx, "first_point:", self.first_point)
                if opp_idx >= len(self.detections):
                    opp_idx -= len(self.detections)

                if self.first_point:
                    self.first_point = False
                    self.prev_time = time.time()
                    self.prev_opp_idx = opp_idx
                else:
                    curr_time = time.time()
                    dt = curr_time - self.prev_time

                    if opp_idx - self.prev_opp_idx < -len(self.detections) / 2:
                        for i in np.arange(self.prev_opp_idx, opp_idx + len(self.detections), 1):
                            self.detections[(i + 1) % len(self.detections)].dt = dt / (opp_idx + len(self.detections) - self.prev_opp_idx)
                            # print("i+1:", (i + 1) % len(self.detections), "opp_idx:", opp_idx, "prev_opp_idx:", self.prev_opp_idx, "dt:", self.detections[(i + 1) % len(self.detections)].dt)
                    else:
                        for i in np.arange(self.prev_opp_idx, opp_idx, 1):
                            self.detections[i + 1].dt = dt / (opp_idx - self.prev_opp_idx)
                            # print("i+1:", i + 1, "opp_idx:", opp_idx, "prev_opp_idx:", self.prev_opp_idx, "dt:", self.detections[i + 1].dt)

                    self.prev_opp_idx = opp_idx
                    self.prev_time = curr_time

                # print("time duration:", msg.dt)
                msg.dt = self.detections[opp_idx].dt
                self.detections[opp_idx] = msg
                self.prev_detection = msg

    def timer_callback(self):
        if not self.done_init:
            return

        marker_array = MarkerArray()
        for i in range(len(self.detections)):
            detection = self.detections[i]
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detected_opp_traj'
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            marker.pose.position.x = detection.x
            marker.pose.position.y = detection.y
            quat = R.from_euler('z', detection.yaw).as_quat()
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]

            marker.scale.x = max(0.1, detection.v * 0.01)
            # marker.scale.x = 0.1
            marker.scale.y = 0.05
            marker.scale.z = 0.05

            marker.color.r = detection.v / 10.0
            marker.color.g = 1.0 - abs(5.0 - detection.v) / 5.0
            marker.color.b = 1.0 - detection.v / 10.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.detected_opp_traj_pub.publish(marker_array)

def main():
    rclpy.init()
    collection = CollectDetection()
    rclpy.spin(collection)
    collection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()