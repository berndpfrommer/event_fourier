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
             'use_sim_time': True,   # attention!
             'frame_id': '',
             'min_frequency': 6.0,
             'max_frequency': 5000.0,
             #'dt_averaging_alpha': 0.1,  # good weight of new period measurement
             'dt_averaging_alpha': 1.0,  # weight of new period measurement
             'prefilter_event_cutoff': 20.0, # prefilter cutoff period #events
             #'reset_threshold': 0.2,  # good
             'reset_threshold': 10000.0,
             'num_timeout_cycles': 3,
             'stale_pixel_threshold': 10.0,
             'debug_x': 298,
             'debug_y': 301,
             'num_frequency_clusters': 0,
             'num_good_cycles_required': 0,
             'use_log_frequency': True,
             'overlay_events': False,
             'legend_width': 0,
             'worker_threads': 0,
             'legend_frequencies': (16.0,  32, 64, 128, 256, 512,  1024, 2048, 4096),
             'use_external_frame_times': True,
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
