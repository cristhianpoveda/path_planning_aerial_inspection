import os
from glob import glob
from setuptools import setup

package_name = 'wildbridge_nav_tests'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cristhian',
    maintainer_email='cristhianpoveda12@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'shape_flyer = wildbridge_nav_tests.shape_flyer_node:main',
            'velocity_odometry = wildbridge_nav_tests.velocity_odometry_node:main'
        ],
    },
)
