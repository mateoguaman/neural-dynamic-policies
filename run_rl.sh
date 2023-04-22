#!/bin/bash


ENV_NAME=$1
TYPE=$2
EXP_ID=$3
SEED=$4
num_steps=150
num_processes=14
DEVICES=$5

cleanup() {
    exit 1
}

if [ "$ENV_NAME" = "throw" ] || [ "$ENV_NAME" = "pick" ]; then
    num_steps=55
    num_processes=38
fi

if [ "$TYPE" = "dmp" ] || [ "$TYPE" = "ppo-multi" ] && [ "$ENV_NAME" = "throw" ]; then
    CUDA_VISIBLE_DEVICES=$DEVICES python ./main_rl.py --env-name $ENV_NAME  --type $TYPE --seed $SEED --run_id $EXP_ID --reward-delay 5 --T 5 --N 5 --a_z 5 --num-processes $num_processes
fi
if [ "$TYPE" = "dmp" ] || [ "$TYPE" = "ppo-multi" ] && [ "$ENV_NAME" = "push" ]; then
    CUDA_VISIBLE_DEVICES=$DEVICES python ./main_rl.py --env-name=$ENV_NAME  --type $TYPE --seed $SEED --run_id $EXP_ID --reward-delay 5 --T 5 --N 6 --a_z 10 --num-processes $num_processes
fi
if [ "$TYPE" = "dmp" ] || [ "$TYPE" = "ppo-multi" ] && [ "$ENV_NAME" = "push" ]; then
    CUDA_VISIBLE_DEVICES=$DEVICES python ./main_rl.py --env-name=$ENV_NAME  --type $TYPE --seed $SEED --run_id $EXP_ID --reward-delay 5 --T 5 --N 6 --a_z 5 --num-processes $num_processes
fi
if [ "$TYPE" = "dmp" ] || [ "$TYPE" = "ppo-multi" ] && [ "$ENV_NAME" = "soccer" ]; then
    CUDA_VISIBLE_DEVICES=$DEVICES python ./main_rl.py --env-name=$ENV_NAME  --type $TYPE --seed $SEED --run_id $EXP_ID --reward-delay 5 --T 5 --N 6 --a_z 15 --num-processes $num_processes
fi
if [ "$TYPE" = "dmp" ] || [ "$TYPE" = "ppo-multi" ] && [ "$ENV_NAME" = "faucet" ]; then
    CUDA_VISIBLE_DEVICES=$DEVICES python ./main_rl.py --env-name=$ENV_NAME  --type $TYPE --seed $SEED --run_id $EXP_ID --reward-delay 5 --T 5 --N 6 --a_z 5 --num-processes $num_processes
fi
if [ "$TYPE" = "ppo" ]; then
    CUDA_VISIBLE_DEVICES=$DEVICES python ./main_rl.py --env-name=$ENV_NAME  --type=$TYPE --seed=$SEED --run_id=$EXP_ID
fi
wait $!
