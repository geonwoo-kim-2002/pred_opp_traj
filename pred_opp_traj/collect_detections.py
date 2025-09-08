import time
import numpy as np
import pandas as pd

from pred_opp_traj.Spline import Spline, Spline2D
from pred_msgs.msg import Detection, DetectionArray

def init_detections(map, pkg_path) -> tuple[DetectionArray, Spline2D]:
    center_path = pd.read_csv(f'{pkg_path}/data/path/{map}_path.csv')
    sp = Spline2D(center_path['x_m'], center_path['y_m'])

    detect_array = DetectionArray()
    try:
        raceline_spline = pd.read_csv(f'{pkg_path}/data/raceline/{map}_race_spline.csv')
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
            detect_array.detections.append(new_detection)

    except:
        print("Raceline spline data not found. Computing raceline spline.", flush=True)
        x_center = []
        y_center = []
        for i in np.arange(0.0, sp.s[-1], 0.1):
            ix, iy = sp.calc_position(i)
            x_center.append(ix)
            y_center.append(iy)

        states = pd.read_csv(f'{pkg_path}/data/raceline/{map}_states.csv')
        t_s = []
        if len(x_center) != len(states['t_s']):
            t_s = np.interp(np.arange(0.0, sp.s[-1], 0.1), states['s_m'], states['t_s'])
        else:
            t_s = states['t_s'].values
        print(f"Length of center path: {len(x_center)}, Length of states data: {len(t_s)}\n", flush=True)

        raceline = pd.read_csv(f'{pkg_path}/data/raceline/{map}_traj_race_cl.csv')
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
            new_detection.x_var = 1.0
            new_detection.y_var = 1.0
            new_detection.yaw_var = 0.5
            new_detection.v_var = 1.0
            detect_array.detections.append(new_detection)

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
        raceline_spline_df.to_csv(f'{pkg_path}/data/raceline/{map}_race_spline.csv', index=False)

    print("Detections initialized.", time.time(), flush=True)
    return detect_array, sp

def get_detection_array(detected_opp: Detection, detect_array: DetectionArray, sp: Spline2D, first_point: bool, prev_time: float, prev_opp_idx: int):
    if time.time() - prev_time >= 0.5:
        first_point = True

    curr_opp_s = sp.find_s(detected_opp.x, detected_opp.y)
    if (curr_opp_s * 100) % 10 <= 2 or (curr_opp_s * 100) % 10 >= 7:
        opp_idx = int(round(curr_opp_s, 1) * 10)
        if opp_idx >= len(detect_array.detections):
            opp_idx -= len(detect_array.detections)

        if first_point:
            first_point = False
            prev_time = time.time()
            prev_opp_idx = opp_idx
        else:
            curr_time = time.time()
            dt = curr_time - prev_time

            if opp_idx - prev_opp_idx < -len(detect_array.detections) / 2:
                for i in np.arange(prev_opp_idx, opp_idx + len(detect_array.detections), 1):
                    detect_array.detections[(i + 1) % len(detect_array.detections)].dt = dt / (opp_idx + len(detect_array.detections) - prev_opp_idx)
            else:
                for i in np.arange(prev_opp_idx, opp_idx, 1):
                    detect_array.detections[i + 1].dt = dt / (opp_idx - prev_opp_idx)

            prev_opp_idx = opp_idx
            prev_time = curr_time

        detected_opp.dt = detect_array.detections[opp_idx].dt
        detect_array.detections[opp_idx] = detected_opp