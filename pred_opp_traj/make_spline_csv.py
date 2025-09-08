import numpy as np
import pandas as pd
from Spline import Spline2D
import matplotlib.pyplot as plt
import math

map = 'map7_reverse'
path = pd.read_csv(f'pred_opp_traj/data/path/{map}_path.csv')
width = pd.read_csv(f'pred_opp_traj/data/path/{map}_width_info.csv')
left_lane = pd.read_csv(f'pred_opp_traj/data/lane/{map}_left.csv')
right_lane = pd.read_csv(f'pred_opp_traj/data/lane/{map}_right.csv')

sp = Spline2D(path['x_m'], path['y_m'])

last_point = sp.calc_position(sp.s[-1] - 1e-10)
first_point = sp.calc_position(sp.s[0])
# print(math.hypot(last_point[0] - first_point[0], last_point[1] - first_point[1]))
print(f"Start: {first_point}, End: {last_point}")
print(f"Length: {sp.s[-1]}")

x = []
y = []
left_width = []
right_width = []
for i in np.arange(0.0, sp.s[-1], 0.1):
    ix, iy = sp.calc_position(i)
    x.append(ix)
    y.append(iy)
    left_width.append(width['left'].values[int(i * 100)])
    right_width.append(width['right'].values[int(i * 100)])
plt.plot(x, y, label='Spline Path', color='blue')
plt.plot(left_lane['x'].values, left_lane['y'].values, color='g')
plt.plot(right_lane['x'].values, right_lane['y'].values, color='g')
plt.show()

header = ("x_m, y_m, w_tr_right_m, w_tr_left_m")
fmt = "%.5f, %.5f, %.5f, %.5f"
np.savetxt(f'pred_opp_traj/data/path/{map}_path_spline.csv', np.column_stack((np.array(x), np.array(y), np.array(right_width), np.array(left_width))), header=header, fmt=fmt)