#!/bin/bash

export PYTHONPATH="$PWD" 

# pos
python hida/run.py \
    --version kappador \
    --dataset_size 512 \
    --train_batch_size 4 \
    --val_batch_size 4 \
    --test_batch_size 4

# pos
python hida/run.py \
    --version dolphin \
    --encoder densenet161

# neg
python hida/run.py \
    --version frog \
    --vflip_chance 0.5

# neg
python hida/run.py \
    --version sysiphus \
    --rotation_chance 0.5

# neg, not trained long enougth
python hida/run.py \
    --version schnappi \
    --learning_rate 0.00001

# neg
python hida/run.py \
    --version octopus \
    --learning_rate 0.001

# neg
python hida/run.py \
    --version fox \
    --architecture pspnet
# 0.128

# pos
python hida/run.py \
    --version fox \
    --architecture pan

# baseline
python hida/run.py \
    --version llama