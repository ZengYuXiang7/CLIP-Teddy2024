#!/bin/bash

# 定义变量
run_name='Experiment'
rounds=5 epochs=300
batch_size=256 learning_rate=0.001 decay=0.001
experiment=1 record=1 program_test=0 verbose=1
dimensions="32 64"
datasets="rt tp"
densities="0.80"
py_files="run_experiment run_ablation"

for py_file in $py_files
do
    for dim in $dimensions
    do
        for dataset in $datasets
        do
            for density in $densities
            do
                python ./$py_file.py \
                      --logger $run_name \
                      --rounds $rounds \
                      --density $density \
                      --dataset $dataset \
                      --epochs $epochs \
                      --bs $batch_size \
                      --lr $learning_rate \
                      --decay $decay \
                      --program_test $program_test \
                      --dimension $dim \
                      --experiment $experiment \
                      --record $record \
                      --verbose $verbose
            done
        done
        # 在命令序列之间添加一个空行
        echo ""
    done
done