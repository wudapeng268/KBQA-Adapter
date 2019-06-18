#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=$1
for i in {0..9}
do
    python -u start.py --run test_kbqa --config $2 --fold $i 
    echo "all done!"
done
