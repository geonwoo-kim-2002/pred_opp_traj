from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os

def get_share_file(*args):
    # return os.path.join(get_package_share_directory(package_name), *args)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), *args)

def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('pred_opp_traj'),
        'params',
        'params.yaml'
        )

    pred_opp_traj_srv = Node(
        package='pred_opp_traj',
        executable='pred_opp_traj_srv',
        name='pred_opp_traj_srv',
        parameters=[config],
        output='screen'
    )
    ld.add_action(pred_opp_traj_srv)
    return ld
