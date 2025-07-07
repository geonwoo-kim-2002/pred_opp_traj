from setuptools import find_packages, setup

package_name = 'pred_opp_traj'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Geon-Woo Kim',
    maintainer_email='apdnxn@gmail.com',
    description='Prediction of Opponent Trajectory in F1TENTH',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_node = pred_opp_traj.detection:main',
        ],
    },
)
