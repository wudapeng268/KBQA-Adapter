#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=$1
config_file="config/mse-all.yml"
config_file2="config/mse-recon-all.yml"
config_file3="config/gan-all.yml"
config_file4="config/gan-recon-all.yml"
start=$2
end=$3
for i in $(seq $start 1 $end)
do
    echo $i
    python -u start.py --run train --config config/baseline.yml --fold $i | tee baseline.$i.log
    python -u start.py --run train --config $config_file --fold $i  | tee $config_file.$i.log
    python -u start.py --run train --config $config_file2 --fold $i  | tee $config_file2.$i.log
    python -u start.py --run train --config $config_file3 --fold $i  | tee $config_file3.$i.log
    python -u start.py --run train --config $config_file4 --fold $i  | tee $config_file4.$i.log
    echo "all done!"
done
