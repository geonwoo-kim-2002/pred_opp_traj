import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import time

from pred_opp_traj.Spline import Spline, Spline2D
from pred_msgs.msg import Detection, DetectionArray

class CollectDetection(Node):
    def __init__(self):
        super().__init__('collect_detection_node')
        self.pkg_path = get_package_share_directory('pred_opp_traj')

        self.declare_parameter('map', '')
        self.done_init = False
        self.detect_array = DetectionArray()
        self.init_detections()

        self.prev_detection = None
        self.first_point = True
        self.prev_time = 0.0
        self.prev_opp_idx = None
        self.detection_sub = self.create_subscription(Detection, '/detection', self.detection_callback, 1)

        self.detection_array_pub = self.create_publisher(DetectionArray, '/detected_opp_traj', 1)
        self.detected_opp_traj_marker_pub = self.create_publisher(MarkerArray, '/detected_opp_traj_marker', 1)
        self.timer = self.create_timer(0.001, self.timer_callback)

    def init_detections(self):
        self.map = self.get_parameter('map').value
        center_path = pd.read_csv(f'{self.pkg_path}/data/path/{self.map}_path.csv')
        self.sp = Spline2D(center_path['x_m'], center_path['y_m'])

        try:
            raceline_spline = pd.read_csv(f'{self.pkg_path}/data/raceline/{self.map}_race_spline.csv')
            print("Using precomputed raceline spline data.", flush=True)

            for _, row in raceline_spline.iterrows():
                new_detection = Detection()
                new_detection.dt = row['dt']
                new_detection.x = row['x']
                new_detection.y = row['y']
                new_detection.yaw = row['yaw']
                new_detection.v = row['v']
                new_detection.x_var = row['x_var']
                new_detection.y_var = row['y_var']
                new_detection.yaw_var = row['yaw_var']
                new_detection.v_var = row['v_var']
                self.detect_array.detections.append(new_detection)

        except:
            print("Raceline spline data not found. Computing raceline spline.", flush=True)
            x_center = []
            y_center = []
            for i in np.arange(0.0, self.sp.s[-1], 0.1):
                ix, iy = self.sp.calc_position(i)
                x_center.append(ix)
                y_center.append(iy)

            states = pd.read_csv(f'{self.pkg_path}/data/raceline/{self.map}_states.csv')
            t_s = []
            if len(x_center) != len(states['t_s']):
                # t_s 보간
                t_s = np.interp(np.arange(0.0, self.sp.s[-1], 0.1), states['s_m'], states['t_s'])
            else:
                t_s = states['t_s'].values
            print(f"Length of center path: {len(x_center)}, Length of states data: {len(t_s)}\n", t_s, flush=True)
            # raise ValueError(f"Length of center path does not match length of states data. spline length: {len(x_center)}, states length: {len(states['t_s'])}")

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

            raceline_spline_list = []
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
                    new_detection.dt = t_s[len(t_s) - 1] - t_s[len(t_s) - 2]
                else:
                    new_detection.dt = t_s[i] - t_s[i - 1]
                new_detection.x = race_x[min_idx]
                new_detection.y = race_y[min_idx]
                new_detection.yaw = race_yaw[min_idx]
                new_detection.v = race_v[min_idx]
                new_detection.x_var = 0.5
                new_detection.y_var = 0.5
                new_detection.yaw_var = 0.1
                new_detection.v_var = 0.5
                self.detect_array.detections.append(new_detection)

                raceline_spline_list.append({
                    'dt': new_detection.dt,
                    'x': new_detection.x,
                    'y': new_detection.y,
                    'yaw': new_detection.yaw,
                    'v': new_detection.v,
                    'x_var': new_detection.x_var,
                    'y_var': new_detection.y_var,
                    'yaw_var': new_detection.yaw_var,
                    'v_var': new_detection.v_var,
                })

            raceline_spline_df = pd.DataFrame(raceline_spline_list)
            raceline_spline_df.to_csv(f'{self.pkg_path}/data/raceline/{self.map}_race_spline.csv', index=False)

        self.done_init = True
        print("Detections initialized.", flush=True)

    def detection_callback(self, msg):
        if time.time() - self.prev_time >= 0.5:
            # print("time duration:", time.time() - self.prev_time, flush=True)
            self.first_point = True

        if msg != self.prev_detection and self.done_init:
            curr_opp_s = self.sp.find_s(msg.x, msg.y)
            if (curr_opp_s * 100) % 10 <= 2 or (curr_opp_s * 100) % 10 >= 7:
                opp_idx = int(round(curr_opp_s, 1) * 10)
                # print("prev_opp_idx:", self.prev_opp_idx, "opp_idx:", opp_idx, "first_point:", self.first_point, flush=True)
                if opp_idx >= len(self.detect_array.detections):
                    opp_idx -= len(self.detect_array.detections)

                if self.first_point:
                    self.first_point = False
                    self.prev_time = time.time()
                    self.prev_opp_idx = opp_idx
                else:
                    curr_time = time.time()
                    dt = curr_time - self.prev_time

                    if opp_idx - self.prev_opp_idx < -len(self.detect_array.detections) / 2:
                        for i in np.arange(self.prev_opp_idx, opp_idx + len(self.detect_array.detections), 1):
                            self.detect_array.detections[(i + 1) % len(self.detect_array.detections)].dt = dt / (opp_idx + len(self.detect_array.detections) - self.prev_opp_idx)
                            # print("i+1:", (i + 1) % len(self.detections), "opp_idx:", opp_idx, "prev_opp_idx:", self.prev_opp_idx, "dt:", self.detections[(i + 1) % len(self.detections)].dt, flush=True)
                    else:
                        for i in np.arange(self.prev_opp_idx, opp_idx, 1):
                            self.detect_array.detections[i + 1].dt = dt / (opp_idx - self.prev_opp_idx)
                            # print("i+1:", i + 1, "opp_idx:", opp_idx, "prev_opp_idx:", self.prev_opp_idx, "dt:", self.detections[i + 1].dt, flush=True)

                    self.prev_opp_idx = opp_idx
                    self.prev_time = curr_time

                # print("time duration:", msg.dt, flush=True)
                msg.dt = self.detect_array.detections[opp_idx].dt
                self.detect_array.detections[opp_idx] = msg
                self.prev_detection = msg

    def timer_callback(self):
        if not self.done_init:
            return

        marker_array = MarkerArray()
        for i in range(len(self.detect_array.detections)):
            if self.detect_array.detections[i].v >= 1.0:
                detection = self.detect_array.detections[i]
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detected_opp_traj'
            marker.id = i
            # marker.type = Marker.ARROW
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = detection.x
            marker.pose.position.y = detection.y
            quat = R.from_euler('z', detection.yaw).as_quat()
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]

            # marker.scale.x = max(0.1, detection.v * 0.01)
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 1e-5

            # marker.color.r = detection.v / 10.0
            # marker.color.g = 1.0 - abs(5.0 - detection.v) / 5.0
            # marker.color.b = 1.0 - detection.v / 10.0
            # marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.detection_array_pub.publish(self.detect_array)
        self.detected_opp_traj_marker_pub.publish(marker_array)

def main():
    rclpy.init()
    collection = CollectDetection()
    rclpy.spin(collection)
    collection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()