from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'avader'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='mietla@student.agh.edu.pl',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hover_start = avader.hover_start:main',
            'camera = avader.camera:main',
            'task_1 = avader.task_1:main',
            'task_2 = avader.task_2:main',
            'task_3 = avader.task_3:main',
        ],
    },
)
