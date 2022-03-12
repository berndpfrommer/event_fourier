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
        executable='event_fourier_node',
        output='screen',
        # prefix=['xterm -e gdb -ex run --args'],
        name='event_fourier',
        parameters=[
            {'use_sensor_time': True,
             'frame_id': '',
             'frequencies': [4.0, 150.0],
#             'roi': [319, 239, 2, 2],
#             'roi': [315, 235, 10, 10],
#             'roi': [300, 220, 40, 40],             
             'roi': [280, 200, 80, 80],
#             'roi': [0, 0, 640, 480],
             'bag_file': LaunchConfig('bag_file').perform(context),
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
        LaunchArg('bag_file', default_value=[''],
                  description='name of bag file to read'),
        OpaqueFunction(function=launch_setup)
        ])
