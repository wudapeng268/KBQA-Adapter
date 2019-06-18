#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=$1

for i in {0..9}
do
    python -u start.py --run train --config config/minus.yml --fold $i | tee minus.log
    python -u start.py --run train --config config/wo.ft.yml --fold $i | tee wo.ft.log
    echo "all done!"
    python send_result.py
done

# bash script/train.sh $1 config/mmm.exp/baseline_forever.yml 0
