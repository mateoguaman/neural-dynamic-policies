DEVICES=$1

echo "Running on GPU $DEVICES"

CUDA_VISIBLE_DEVICES=$DEVICES python main_il.py