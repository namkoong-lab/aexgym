#!/bin/bash

EXP_ID_LIST_PATH='scripts/conf/exp_ids.txt' 
SESSION_NAME="MixPolicies"
if [ ! -f "$EXP_ID_LIST_PATH" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Attaching to it."
    tmux attach-session -t $SESSION_NAME
else
    # Create a new session and detach from it
    tmux new-session -d -s $SESSION_NAME
fi 

while IFS= read -r line
do
    # Create a new window in tmux for each command
    # The window will be named after the command
    tmux new-window -t $SESSION_NAME -n "$line"
    tmux send-keys -t $SESSION_NAME:"$line" "conda activate jwenv-01" C-m
    tmux send-keys -t $SESSION_NAME:"$line" "python3 lr_search.py exp_id=$line" C-m
    echo $line 
done < "$EXP_ID_LIST_PATH"
