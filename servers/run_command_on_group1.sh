#!/bin/bash

remote_group="group1"
remote_servers=( \
    "job-9e634f57-9680-4789-9e37-63c68a73f40f-master-0" \
    "job-55d9b73e-905a-49de-b2ee-4e04d266bf40-master-0" \
    "job-bca53fac-37ff-4b04-8fec-fab04e526787-master-0" \
    "job-144eb281-171d-42bd-ba35-f138ced874e0-master-0" \
)

# remote_command="bash train.sh"
remote_command="pkill -f torch && pkill -f main.py && pkill -f wandb"


envstr="export MASTER_ADDR=${remote_servers[0]}; MASTER_PORT=23456; export WORLD_SIZE=${#remote_servers[@]}"
while IFS= read -r line
do
  varname=$(echo "$line" | cut -d= -f1)
  varvalue=$(echo "$line" | cut -d= -f2)
  if [ "$varname" != "RANK" ] && [ "$varname" != "WORLD_SIZE" ]; then
    envstr="$envstr; export $varname=\"$varvalue\""
  fi
done < servers/$remote_group/env.txt

rm -rf servers/$remote_group/*.log
sleep 2s
for ((i=0; i<${#remote_servers[@]}; i++))
do
    server="${remote_servers[i]}"
    echo "Connecting to $server..."
    setenv_command="$envstr;export RANK=$i"
    echo $remote_command
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$server "$setenv_command;cd $(pwd);nohup $remote_command > servers/$remote_group/rank$i.log 2>&1 &" &
    if [ $? -eq 0 ]; then
        echo "Command started on $server."
    else
        echo "Failed to connect to $server."
    fi
done
