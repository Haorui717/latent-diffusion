#!/bin/bash

config_root="/home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Local/configs"
config_paths=(
    "$config_root/haorui/ldm_onlymask/ldm_onlymask_1800.yaml"
)
output_root="/home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Local/tmux_log"
output_paths=(
    "$output_root/ldm_onlymask/ldm_onlymask_1800_$(date +"%Y:%m:%d:%H:%M:%S").txt"
)

# An array of commands you want to run
declare -a commands=(
    "conda activate ldm; python main.py --base ${config_paths[0]} -t --logdir logs/ldm_onlymask --gpu 0,1,2,3,4,5,6,7 2>&1 | tee ${output_paths[0]}"
)

declare -a session_names=(
    "ldm_onlymask_1800"
)

# Loop through the indices of the commands array
for index in "${!commands[@]}"; do
    cmd="${commands[$index]}"
    session_name="${session_names[$index]}"
    
    # Create a new detached tmux session and run the command
    tmux new-session -d -s $session_name "bash"
    tmux send-keys -t $session_name "$cmd" C-m
    
    # Sleep for a short while to space out the start of each command (optional)
    sleep 1
done

echo "All training sessions started in tmux."