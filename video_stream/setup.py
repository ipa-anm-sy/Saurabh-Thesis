import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'video_stream'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='server',
    maintainer_email='saurabh.yeola@ipa.fraunhofer.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'stream = video_stream.video_stream:main',
        'facedetection = video_stream.face_detect:main',
        'pattlite= video_stream.pattlite:main',
        'rafdb_single= video_stream.rafdb_single:main',
        'facetracking=video_stream.pred_facetracking:main',
        'yolopredict=video_stream.yolo_emo_pred:main',
        'yoloface=video_stream.yoloface:main',
        'yolotrack=video_stream.yolotrack:main',
        
        ],
    },
)
