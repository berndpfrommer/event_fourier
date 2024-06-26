#
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

cmake_minimum_required(VERSION 3.5)
project(event_fourier)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_ros REQUIRED)

find_package(OpenCV REQUIRED)

set(ROS2_DEPENDENCIES
  "event_camera_msgs"
  "event_camera_codecs"
  "rclcpp"
  "rclcpp_components"
  "rosbag2_cpp"
  "image_transport"
  "cv_bridge"
)

foreach(pkg ${ROS2_DEPENDENCIES})
  find_package(${pkg} REQUIRED)
endforeach()

if(${cv_bridge_VERSION} VERSION_GREATER "3.3.0")
  message(WARNING "using hpp")
  add_definitions(-DUSE_CV_BRIDGE_HPP)
endif()

ament_auto_find_build_dependencies(REQUIRED ${ROS2_DEPENDENCIES})

#
# ---- event_fourier shared library/component
#
ament_auto_add_library(event_fourier SHARED
  src/event_fourier.cpp
  )

rclcpp_components_register_nodes(event_fourier "event_fourier::EventFourier")

ament_auto_add_executable(event_fourier_node
  src/event_fourier_node.cpp)

#
# ---- frequency_cam shared library/component
#
ament_auto_add_library(frequency_cam SHARED
  src/frequency_cam.cpp
  )

rclcpp_components_register_nodes(frequency_cam "event_fourier::FrequencyCam")

ament_auto_add_executable(frequency_cam_node
  src/frequency_cam_node.cpp
  )


# -------- installation

# the shared library goes into the global lib dir so it can
# be used as a composable node by other projects

install(TARGETS
  event_fourier
  frequency_cam
  DESTINATION lib
)

# the node must go into the paroject specific lib directory or else
# the launch file will not find it

install(TARGETS
  event_fourier_node
  frequency_cam_node
  DESTINATION lib/${PROJECT_NAME}/)

# install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
  FILES_MATCHING PATTERN "*.py")


if(BUILD_TESTING)
  find_package(ament_cmake REQUIRED)
  find_package(ament_cmake_copyright REQUIRED)
  find_package(ament_cmake_cppcheck REQUIRED)
  find_package(ament_cmake_cpplint REQUIRED)
  find_package(ament_cmake_clang_format REQUIRED)
  find_package(ament_cmake_flake8 REQUIRED)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_pep257 REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)

  ament_copyright()
  ament_cppcheck(LANGUAGE c++)
  ament_cpplint(FILTERS "-build/include,-runtime/indentation_namespace")
  ament_clang_format()
  ament_flake8()
  ament_lint_cmake()
  ament_pep257()
  ament_xmllint()
endif()


ament_package()
