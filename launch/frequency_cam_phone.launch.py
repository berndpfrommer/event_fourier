# -----------------------------------------------------------------------------
# Copyright 2022 Bernd Pfrommer <bernd.pfrommer@gmail.com>
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
        name='frequency_cam',
        parameters=[
            {'use_sensor_time': True,
             'use_sim_time': False,   # attention!
             'frame_id': '',
             'min_frequency': 50.0,
             'max_frequency': 250.0,
             'dt_averaging_alpha': 0.5,  # weight of new period measurement
             'prefilter_event_cutoff': 2.0, # prefilter cutoff period #events
             'reset_threshold': 0.2,
             'debug_x': 319,
             'debug_y': 239,
             'num_frequency_clusters': 0,
             'use_log_frequency': False,
             'overlay_events': False,
             'worker_threads': 0,
             # 'legend_frequencies': (180.0, 200, 220, 240, 260, 280, 300),
             # open tuning:
             # d-2   73.4
             # a-2  110.0
             # d-3  146.8   -> g-3 196.0
             # f#-3 185.0   -> e-3 164.8
             # a-3  220.0   -> b-3 246.9
             # d-3  293.7
        
#             'legend_frequencies': (73.4, 110.0, 146.8, 185.0, 220.0, 293.7),
#             'roi': [319, 239, 2, 2],
#             'roi': [315, 235, 10, 10],
#             'roi': [300, 220, 40, 40],             
#             'roi': [280, 200, 80, 80],
#             'roi': [0, 0, 640, 480],
             'bag_file': LaunchConfig('bag').perform(context),
             'publishing_frequency': 25.0}],
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
