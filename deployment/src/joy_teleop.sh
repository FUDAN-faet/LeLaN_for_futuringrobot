#!/bin/bash

# Create a new tmux session
session_name="teleop_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into two panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

# Run the ROS 2 launch file in the first pane
tmux select-pane -t 0
tmux send-keys "ros2 launch lelan_deployment vint_locobot.launch.py" Enter

# Run the teleop node in the second pane
tmux select-pane -t 1
tmux send-keys "ros2 run lelan_deployment joy_teleop" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
