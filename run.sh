#!/bin/bash

export PYTHONPATH="$PWD" 

python hida/run.py \
    --version kappador \
    --dataset_size 512 \
    --train_batch_size 4 \
    --val_batch_size 4 \
    --test_batch_size 4

python hida/run.py \
    --version dolphin \
    --encoder densenet161

python hida/run.py \
    --version frog \
    --vflip_chance 0.5

python hida/run.py \
    --version sysiphus \
    --rotation_chance 0.5

python hida/run.py \
    --version schnappi \
    --learning_rate 0.00001

python hida/run.py \
    --version octopus \
    --learning_rate 0.001

python hida/run.py \
    --version fox \
    --architecture pspnet

python hida/run.py \
    --version fox \
    --architecture pan

python hida/run.py \
    --version llama