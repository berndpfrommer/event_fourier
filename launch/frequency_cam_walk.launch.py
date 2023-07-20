# -----------------------------------------------------------------------------
# Copyright 2021 Bernd Pfrommer <bernd.pfrommer@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

import launch
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration as LaunchConfig
from launch.actions import DeclareLaunchArgument as LaunchArg
from launch.actions import OpaqueFunction


def launch_setup(context, *args, **kwargs):
    """function to set up launch."""
    image_topic_config = LaunchConfig('image_topic')
    event_topic_config = LaunchConfig('event_topic')
    image_topic = image_topic_config.perform(context)
    event_topic = event_topic_config.perform(context)
    node = Node(
        package='event_fourier',
        executable='frequency_cam_node',
        output='screen',
        # prefix=['xterm -e gdb catch throw -ex run --args'],
        name='frequency_cam',
        parameters=[
            {'use_sensor_time': True,
             'frame_id': '',
             'min_frequency': 1.0,
             'max_frequency': 1000.0,
             'dt_averaging_alpha': 0.99,  # weight of new period measurement
             'prefilter_event_cutoff': 40.0, # prefilter cutoff period #events
             'reset_threshold': 5.0,
             'debug_x': 319,
             'debug_y': 239,
             'num_frequency_clusters': 5,
             'use_log_frequency': True,
             'overlay_events': True,
             'worker_threads': 0,
             # 'legend_frequencies': (180.0, 200, 220, 240, 260, 280, 300),
#             'roi': [319, 239, 2, 2],
#             'roi': [315, 235, 10, 10],
#             'roi': [300, 220, 40, 40],             
#             'roi': [280, 200, 80, 80],
#             'roi': [0, 0, 640, 480],
             'bag_file': LaunchConfig('bag').perform(context),
             'slice_time': 0.03}],
        remappings=[
            ('~/events', event_topic),
            ('~/image', image_topic)
        ])
    return [node]


def generate_launch_description():
    """Create slicer node by calling opaque function."""
    return launch.LaunchDescription([
        LaunchArg('image_topic', default_value=['/event_camera/image'],
                  description='image topic'),
        LaunchArg('event_topic', default_value=['/event_camera/events'],
                  description='event topic'),
        LaunchArg('bag', default_value=[''],
                  description='name of bag file to read'),
        OpaqueFunction(function=launch_setup)
        ])
