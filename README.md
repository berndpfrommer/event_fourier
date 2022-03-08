# event_frontend

This repository has ROS2 based fourier analysis code for event based cameras.

## Supported platforms

Currently tested on Ubuntu 20.04 under ROS2 Galactic.


## How to build
Create a workspace (``event_fourier``), clone this repo, and use ``wstool``
to pull in the remaining dependencies:

```
mkdir -p ~/event_fourier/src
cd ~/event_fourier
git clone https://github.com/berndpfrommer/event_fourier.git src/event_fourier
wstool init src src/event_fourier/event_fourier.rosinstall
# to update an existing space:
# wstool merge -t src src/event_fourier/event_fourier.rosinstall
# wstool update -t src
```

### configure and build:

```
cd ~/event_fourier
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo  # (optionally add -DCMAKE_EXPORT_COMPILE_COMMANDS=1)
```

## How to use (ROS2):
```
ros2 launch event_fourier event_fourier.launch.py
```

## License

This software is issued under the Apache License Version 2.0.
