#!/bin/bash

# Create a new tmux session
session_name="gnm_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into two panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves

# Run the create_topomap node in the first pane
tmux select-pane -t 0
tmux send-keys "ros2 run lelan_deployment create_topomap --dt 1 --dir $1" Enter

# Change the directory to ../topomaps/bags and run ros2 bag play in the second pane
tmux select-pane -t 1
tmux send-keys "mkdir -p ../topomaps/bags" Enter
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "ros2 bag play $2 --rate 1.5" # feel free to change the playback rate to change the edge length in the graph

# Attach to the tmux session
tmux -2 attach-session -t $session_name
