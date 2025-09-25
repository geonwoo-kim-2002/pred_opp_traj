## Getting started
**Install Dependencies**
```bash
cd ~/Downloads && git clone https://github.com/geonwoo-kim-2002/libgp.git
cd libgp && cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && sudo make install
```
</br>

**Install messages**
```bash
cd ~/<your_workspace>/src
git clone https://github.com/geonwoo-kim-2002/f1_msgs.git
git clone https://github.com/geonwoo-kim-2002/pred_msgs.git
```

## Getting Started
```bash
cd ~/<your_workspace>
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 launch pred_opp_traj pred_opp_traj.launch.py
```
</br>