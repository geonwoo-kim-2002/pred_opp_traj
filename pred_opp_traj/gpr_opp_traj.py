import copy
import numpy as np
from math import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

from pred_opp_traj.Spline import Spline2D
from pred_msgs.msg import Detection, DetectionArray

def gpr_pred_opp_traj(curr_opp: Detection, detection_array: DetectionArray, horizon: int, dt: float, sp: Spline2D) -> DetectionArray:
    pred_opp_traj = DetectionArray()
    if curr_opp.v < 0.5:
        for i in range(horizon):
            detection = Detection()
            detection.dt = i * dt
            detection.x = curr_opp.x
            detection.y = curr_opp.y
            detection.yaw = curr_opp.yaw
            detection.v = curr_opp.v
            detection.x_var = 0.05
            detection.y_var = 0.05
            detection.yaw_var = 0.05
            detection.v_var = 0.05
            pred_opp_traj.detections.append(detection)
    else:
        curr_opp_s = sp.find_s(curr_opp.x, curr_opp.y)
        back_opp_idx = int(floor(curr_opp_s * 10)) % len(detection_array.detections)
        front_opp_idx = int(ceil(curr_opp_s * 10)) % len(detection_array.detections)

        da_copy = copy.deepcopy(detection_array.detections)
        sorted_detection_array = DetectionArray()
        for i in range(int((horizon + 5) * dt * 10) * 10):
            idx = (front_opp_idx + i) % len(da_copy)
            if idx == front_opp_idx:
                da_copy[idx].dt = (1 - (curr_opp_s * 10 - back_opp_idx)) * da_copy[idx].dt
            else:
                da_copy[idx].dt = da_copy[idx - 1].dt + da_copy[idx].dt
            sorted_detection_array.detections.append(da_copy[idx])

        x_var = np.hstack((np.array([0.01]), np.array([d.x_var for d in sorted_detection_array.detections])))
        y_var = np.hstack((np.array([0.01]), np.array([d.y_var for d in sorted_detection_array.detections])))
        yaw_var = np.hstack((np.array([0.01]), np.array([d.yaw_var for d in sorted_detection_array.detections])))
        v_var = np.hstack((np.array([0.01]), np.array([d.v_var for d in sorted_detection_array.detections])))
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

        gp_x.fit(np.hstack((np.array([0.0]), np.array([d.dt for d in sorted_detection_array.detections]))).reshape(-1, 1), np.hstack((np.array([curr_opp.x]), np.array([d.x for d in sorted_detection_array.detections]))))
        gp_y.fit(np.hstack((np.array([0.0]), np.array([d.dt for d in sorted_detection_array.detections]))).reshape(-1, 1), np.hstack((np.array([curr_opp.y]), np.array([d.y for d in sorted_detection_array.detections]))))
        gp_yaw.fit(np.hstack((np.array([0.0]), np.array([d.dt for d in sorted_detection_array.detections]))).reshape(-1, 1), np.hstack((np.array([curr_opp.yaw]), np.array([d.yaw for d in sorted_detection_array.detections]))))
        gp_v.fit(np.hstack((np.array([0.0]), np.array([d.dt for d in sorted_detection_array.detections]))).reshape(-1, 1), np.hstack((np.array([curr_opp.v]), np.array([d.v for d in sorted_detection_array.detections]))))

        pred_time = np.arange(0, horizon * dt, dt)
        pred_x, x_std = gp_x.predict(pred_time.reshape(-1, 1), return_std=True)
        pred_y, y_std = gp_y.predict(pred_time.reshape(-1, 1), return_std=True)
        pred_yaw, yaw_std = gp_yaw.predict(pred_time.reshape(-1, 1), return_std=True)
        pred_v, v_std = gp_v.predict(pred_time.reshape(-1, 1), return_std=True)

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
            pred_opp_traj.detections.append(detection)
    return pred_opp_traj