#!/bin/bash

# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves

# Run the ROS 2 launch file in the first pane
tmux select-pane -t 0
tmux send-keys "ros2 launch lelan_deployment vint_locobot.launch.py" Enter

# Run the teleop node in the second pane
tmux select-pane -t 1
tmux send-keys "ros2 run lelan_deployment joy_teleop" Enter

# Change the directory to ../topomaps/bags and run ros2 bag record in the third pane
tmux select-pane -t 2
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "ros2 bag record /usb_cam/image_raw -o $1" # change topic if necessary

# Attach to the tmux session
tmux -2 attach-session -t $session_name
