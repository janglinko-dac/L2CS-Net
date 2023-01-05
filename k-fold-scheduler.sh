#!/bin/bash
# set a path to venv on the machine
ENV_PATH="/home/czarek/envs/web-eye-tracking-algo"
DIRECTORY_PATH="/home/czarek/Documents/work/web-eye-tracking-algo/code/third_party/L2CS-Net"
DATA_PATH="/home/czarek/Downloads/demo_dataset_3/train"
activate () {
    . $ENV_PATH/bin/activate
}

activate

# set the parameters of the k-fold training
python $DIRECTORY_PATH/clear_kfold_hydrant_training.py \
# --train-dir $DATA_PATH \
--architecture efficientnet_b3 \
--batch-size 8 \
--clearml-experiment local_experimet_01 \
--clearml-tags kfold local test scheduled \
--epochs 5 \
--k-fold 5 \
--lr 1e-05 \
--pitch-cls-scale 1.0 \
--pitch-lower-range -42 \
--pitch-reg-scale 1.0 \
--pitch-resolution 3 \
--pitch-upper-range 42 \
--smoothing-sigma 0.45 \
--smoothing-threshold 0.001 \
--yaw-cls-scale 1.0 \
--yaw-lower-range -42 \
--yaw-reg-scale 1.0 \
--yaw-resolution 3 \
--yaw-upper-range 42

deactivate