#!/usr/bin/env bash


prefix="config"
echo $prefix
export CUDA_VISIBLE_DEVICES=$1
start=$2
end=$3
for i in $(seq $start 100 $end)
do
    echo $i
    python -u start.py --run test --config $prefix"/baseline.yml" --fold 3 --same_tl --key_num $i| tee baseline-tl-$i.log
    python convert_result.py --config $prefix"/baseline.yml" --key_num $i --same_tl

    python -u start.py --run test --config $prefix"/gan-recon-all.yml" --fold 3 --same_tl --key_num $i| tee gan-recon-tl-$i.log
    python convert_result.py --config $prefix"/gan-recon-all.yml" --key_num $i --same_tl
done
echo "all done!"