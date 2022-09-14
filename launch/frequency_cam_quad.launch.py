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
             'use_sim_time': True,   # attention!
             'frame_id': '',
             'min_frequency': 200.0,
             'max_frequency': 300.0,
             #'dt_averaging_alpha': 0.2,  # weight of new period measurement
             'dt_averaging_alpha': 1.0,  # weight of new period measurement
             #'num_good_cycles_required': 1, # keep low to get good response
             'num_good_cycles_required': 1, # keep low to get good response
             'prefilter_event_cutoff': 5.0, # prefilter cutoff period #events
             #'reset_threshold': 0.15,
             'reset_threshold': 0.2,
             'debug_x': 148,
             'debug_y': 362,
             'num_frequency_clusters': 0,
             'use_log_frequency': False,
             'overlay_events': True,
             'legend_width': 0,
             'worker_threads': 0,
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
