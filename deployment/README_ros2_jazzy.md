# ROS 2 Jazzy Deployment

This package adds a ROS 2 Jazzy deployment path for the models in this repository.
The training code remains a regular Python package; only the deployment layer is turned
into an `ament_python` package.

## Prerequisites

Install the ROS 2 side packages you need on Ubuntu with Jazzy sourced:

```bash
sudo apt install \
  ros-jazzy-cv-bridge \
  ros-jazzy-joy \
  ros-jazzy-usb-cam
```

Install the Python-side model dependencies the same way as the original project:

```bash
pip install -e train/
```

If `vint_train` is not installed into the active environment, set:

```bash
export LELAN_TRAIN_ROOT=/abs/path/to/learning-language-navigation/train
```

## Build

From the repository root:

```bash
colcon build --packages-select lelan_deployment --symlink-install
source install/setup.bash
```

## Bringup

Launch the ROS 2 camera, joystick, and command mux nodes:

```bash
ros2 launch lelan_deployment vint_locobot.launch.py
```

The launch file keeps the camera under the `usb_cam` namespace, so the default image topic
is still `/usb_cam/image_raw`.

## Main Commands

Teleop:

```bash
ros2 run lelan_deployment joy_teleop
```

Graph navigation:

```bash
ros2 run lelan_deployment pd_controller
ros2 run lelan_deployment navigate --dir topomap --model vint
```

LeLaN final approach:

```bash
ros2 run lelan_deployment lelan_policy_col --prompt "office chair"
```

Long-distance navigation with final LeLaN handoff:

```bash
ros2 run lelan_deployment pd_controller_lelan
ros2 run lelan_deployment navigate_lelan --prompt "office chair" --dir topomap --model vint
```

Topomap creation from a ROS 2 bag:

```bash
ros2 run lelan_deployment create_topomap --dir topomap --dt 1
ros2 bag play path/to/bag --rate 1.5
```

## Notes

- The original ROS 1 scripts in `deployment/src/` are kept, but the ROS 2 entrypoints live in
  `deployment/lelan_deployment/`.
- The tmux helper scripts under `deployment/src/*.sh` were updated to call `ros2 launch`,
  `ros2 run`, and `ros2 bag`.
- The new `cmd_vel_mux` node reproduces the old prioritized velocity mux behavior using
  `deployment/config/cmd_vel_mux.yaml`, so you do not need `yocs_cmd_vel_mux` on ROS 2.
