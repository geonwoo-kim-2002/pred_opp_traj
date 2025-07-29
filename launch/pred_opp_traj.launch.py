from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('pred_opp_traj'),
        'params',
        'params.yaml'
        )

    detection_node = Node(
        package='pred_opp_traj',
        executable='detection_node',
        name='detection_node',
        parameters=[config],
    )
    collect_detection_node = Node(
        package='pred_opp_traj',
        executable='collect_detection_node',
        name='collect_detection_node',
        parameters=[config],
    )
    gpr_opp_traj_node = Node(
        package='pred_opp_traj',
        executable='gpr_opp_traj_node',
        name='gpr_opp_traj_node',
        parameters=[config],
    )
    ld.add_action(detection_node)
    ld.add_action(collect_detection_node)
    ld.add_action(gpr_opp_traj_node)
    return ld
