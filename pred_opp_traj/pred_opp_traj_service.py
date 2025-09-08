import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import time
from math import *
# from transforms3d import euler
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from tf2_ros import TransformBroadcaster

from pred_msgs.msg import Detection, DetectionArray
from rl_switching_mpc_srv.srv import PredOppTraj

from pred_opp_traj.detection import detection
from pred_opp_traj.collect_detections import init_detections, get_detection_array
from pred_opp_traj.gpr_opp_traj import gpr_pred_opp_traj
from pred_opp_traj.Spline import Spline, Spline2D
from visualization_msgs.msg import Marker, MarkerArray

class PredOppTrajService(Node):
    def __init__(self):
        super().__init__('pred_opp_traj_service')
        self.declare_parameter('map', '')
        self.declare_parameter('horizon', 0)
        self.declare_parameter('dt', 0.0)

        self.map = self.get_parameter('map').value
        self.horizon = self.get_parameter('horizon').value
        self.dt = self.get_parameter('dt').value
        self.pkg_path = get_package_share_directory('pred_opp_traj')

        self.scan_pub = self.create_publisher(LaserScan, '/scan_srv', 10)
        self.detect_marker_pub = self.create_publisher(Marker, '/detection_marker', 10)
        self.detected_opp_traj_marker_pub = self.create_publisher(MarkerArray, '/detected_opp_traj_marker', 10)
        self.pred_opp_traj_marker_pub = self.create_publisher(MarkerArray, '/pred_opp_traj_marker', 10)

        print(f"map: {self.map}, horizon: {self.horizon}, dt: {self.dt}", flush=True)
        self.detect_array, self.sp = init_detections(self.map, self.pkg_path)
        marker_array = MarkerArray()
        for i in range(len(self.detect_array.detections)):
            detected = self.detect_array.detections[i]
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detected_opp_traj'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = detected.x
            marker.pose.position.y = detected.y
            quat = R.from_euler('z', detected.yaw).as_quat()
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]

            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 1e-5
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)
        self.detected_opp_traj_marker_pub.publish(marker_array)
        self.prev_time = 0.0
        self.first_point = True
        self.prev_opp_idx = None

        self.br = TransformBroadcaster(self)
        self.srv = self.create_service(PredOppTraj, 'pred_opp_trajectory', self.pred_opp_trajectory_callback)

    def _publish_laser_transforms(self, ts):
        ego_scan_ts = TransformStamped()
        ego_scan_ts.transform.translation.x = 0.0
        ego_scan_ts.transform.rotation.w = 1.
        ego_scan_ts.header.stamp = ts
        ego_scan_ts.header.frame_id = 'ego_racecar' + '/base_link1'
        ego_scan_ts.child_frame_id = 'ego_racecar' + '/laser'
        self.br.sendTransform(ego_scan_ts)

    def pred_opp_trajectory_callback(self, request, response):
        curr_time = time.time()
        ts = self.get_clock().now().to_msg()
        request.scan.header.frame_id = 'ego_racecar/laser'
        request.scan.header.stamp = ts
        self.scan_pub.publish(request.scan)

        if request.reset_collections:
            self.detect_array, self.sp = init_detections(self.map, self.pkg_path)

        ego_t = Transform()
        ego_t.rotation.x = request.ego_odom.pose.pose.orientation.x
        ego_t.rotation.y = request.ego_odom.pose.pose.orientation.y
        ego_t.rotation.z = request.ego_odom.pose.pose.orientation.z
        ego_t.rotation.w = request.ego_odom.pose.pose.orientation.w
        ego_t.translation.x = request.ego_odom.pose.pose.position.x
        ego_t.translation.y = request.ego_odom.pose.pose.position.y
        ego_t.translation.z = 0.0

        ego_ts = TransformStamped()
        ego_ts.transform = ego_t
        ego_ts.header.stamp = ts
        ego_ts.header.frame_id = 'map'
        ego_ts.child_frame_id = 'ego_racecar' + '/base_link1'
        self.br.sendTransform(ego_ts)
        self._publish_laser_transforms(ts)

        detected_opp = detection(request.scan, request.ego_odom, request.opp_odom)
        # if detected_opp is not None:
        #     marker = Marker()
        #     marker.header.frame_id = "map"
        #     marker.header.stamp = self.get_clock().now().to_msg()
        #     marker.id = 0
        #     marker.type = Marker.ARROW
        #     marker.action = Marker.ADD
        #     marker.pose.position.x = detected_opp.x
        #     marker.pose.position.y = detected_opp.y
        #     quat = R.from_euler('z', detected_opp.yaw).as_quat()
        #     marker.pose.orientation.x = quat[0]
        #     marker.pose.orientation.y = quat[1]
        #     marker.pose.orientation.z = quat[2]
        #     marker.pose.orientation.w = quat[3]
        #     marker.scale.x = detected_opp.v * 0.2
        #     marker.scale.y = 0.2
        #     marker.scale.z = 0.2
        #     marker.color.r = 1.0
        #     marker.color.g = 0.0
        #     marker.color.b = 0.0
        #     marker.color.a = 1.0
        #     self.detect_marker_pub.publish(marker)

        if detected_opp is not None:
            get_detection_array(detected_opp, self.detect_array, self.sp, self.first_point, self.prev_time, self.prev_opp_idx)

            # marker_array = MarkerArray()
            # for i in range(len(self.detect_array.detections)):
            #     detected = self.detect_array.detections[i]
            #     marker = Marker()
            #     marker.header.frame_id = 'map'
            #     marker.header.stamp = self.get_clock().now().to_msg()
            #     marker.ns = 'detected_opp_traj'
            #     marker.id = i
            #     marker.type = Marker.SPHERE
            #     marker.action = Marker.ADD

            #     marker.pose.position.x = detected.x
            #     marker.pose.position.y = detected.y
            #     quat = R.from_euler('z', detected.yaw).as_quat()
            #     marker.pose.orientation.x = quat[0]
            #     marker.pose.orientation.y = quat[1]
            #     marker.pose.orientation.z = quat[2]
            #     marker.pose.orientation.w = quat[3]

            #     marker.scale.x = 0.05
            #     marker.scale.y = 0.05
            #     marker.scale.z = 1e-5
            #     marker.color.r = 0.0
            #     marker.color.g = 1.0
            #     marker.color.b = 0.0
            #     marker.color.a = 1.0

            #     marker_array.markers.append(marker)
            # self.detected_opp_traj_marker_pub.publish(marker_array)

        if detected_opp is not None and self.detect_array is not None:
            ego_s = self.sp.find_s(request.ego_odom.pose.pose.position.x, request.ego_odom.pose.pose.position.y)
            opp_s = self.sp.find_s(detected_opp.x, detected_opp.y)
            if opp_s - ego_s < -self.sp.s[-1] / 2:
                opp_s += self.sp.s[-1]
            elif opp_s - ego_s > self.sp.s[-1] / 2:
                opp_s -= self.sp.s[-1]

            if abs(opp_s - ego_s) <= 7.0:
                pred_opp_traj = gpr_pred_opp_traj(detected_opp, self.detect_array, self.horizon, self.dt, self.sp)
                response.pred_opp_traj = pred_opp_traj
                # print(f"Predicted {len(pred_opp_traj.detections)} future detections.", flush=True)
                # marker_array = MarkerArray()
                # for i in range(len(pred_opp_traj.detections)):
                #     marker = Marker()
                #     marker.header.frame_id = 'map'
                #     marker.header.stamp = self.get_clock().now().to_msg()
                #     marker.ns = 'pred_opp_traj'
                #     marker.id = i
                #     marker.type = Marker.SPHERE
                #     marker.action = Marker.ADD

                #     marker.pose.position.x = pred_opp_traj.detections[i].x
                #     marker.pose.position.y = pred_opp_traj.detections[i].y
                #     quat = R.from_euler('z', pred_opp_traj.detections[i].yaw).as_quat()
                #     marker.pose.orientation.x = quat[0]
                #     marker.pose.orientation.y = quat[1]
                #     marker.pose.orientation.z = quat[2]
                #     marker.pose.orientation.w = quat[3]
                #     marker.pose.orientation.w = 1.0

                #     marker.scale.x = 0.1
                #     marker.scale.y = 0.1
                #     marker.scale.z = 1e-5
                #     marker.color.r = pred_opp_traj.detections[i].v / 10.0
                #     marker.color.g = 0.0
                #     marker.color.b = -(pred_opp_traj.detections[i].v - 10.0) / 10.0
                #     marker.color.a = 1.0
                #     marker_array.markers.append(marker)

                #     marker = Marker()
                #     marker.header.frame_id = 'map'
                #     marker.header.stamp = self.get_clock().now().to_msg()
                #     marker.ns = 'pred_opp_traj'
                #     marker.id = i + 100
                #     marker.type = Marker.SPHERE
                #     marker.action = Marker.ADD

                #     marker.pose.position.x = pred_opp_traj.detections[i].x
                #     marker.pose.position.y = pred_opp_traj.detections[i].y
                #     marker.pose.orientation.w = 1.0

                #     marker.scale.x = pred_opp_traj.detections[i].x_var * 2
                #     marker.scale.y = pred_opp_traj.detections[i].y_var * 2
                #     marker.scale.z = 0.0

                #     marker.color.r = 0.0
                #     marker.color.g = 0.0
                #     marker.color.b = 0.0
                #     marker.color.a = 0.1
                #     marker_array.markers.append(marker)
                # self.pred_opp_traj_marker_pub.publish(marker_array)
            else:
                response.pred_opp_traj = DetectionArray()
        else:
            response.pred_opp_traj = DetectionArray()
        # print(f'Predicted Opponent Trajectory Service called in {time.time() - curr_time:.2f} seconds.', flush=True)
        return response

def main():
    rclpy.init()
    service = PredOppTrajService()
    rclpy.spin(service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()