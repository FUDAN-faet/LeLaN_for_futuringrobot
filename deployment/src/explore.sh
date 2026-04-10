#!/bin/bash

# Create a new tmux session
session_name="vint_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane

# Run the ROS 2 launch file in the first pane
tmux select-pane -t 0
tmux send-keys "ros2 launch lelan_deployment vint_locobot.launch.py" Enter

# Run the explore node with command line args in the second pane
tmux select-pane -t 1
tmux send-keys "ros2 run lelan_deployment explore $@" Enter

# Run the teleop node in the third pane
tmux select-pane -t 2
tmux send-keys "ros2 run lelan_deployment joy_teleop" Enter

# Run the PD controller in the fourth pane
tmux select-pane -t 3
tmux send-keys "ros2 run lelan_deployment pd_controller" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
