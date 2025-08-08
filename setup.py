from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pred_opp_traj'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'data', 'lane'), glob('data/lane/*')),
        (os.path.join('share', package_name, 'data', 'path'), glob('data/path/*')),
        (os.path.join('share', package_name, 'data', 'raceline'), glob('data/raceline/*')),
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Geon-Woo Kim',
    maintainer_email='apdnxn@gmail.com',
    description='Prediction of Opponent Trajectory in F1TENTH',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pred_opp_traj_service = pred_opp_traj.pred_opp_traj_service:main',
        ],
    },
)
