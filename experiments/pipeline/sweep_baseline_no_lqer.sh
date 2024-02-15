#!/bin/bash

if [ -z $1 ]; then
    echo "! Requires argument <config_template>"
    exit 1
fi
config_template=$1

if [ -z $2 ]; then
    echo "! Requires argument <tag>, such as model name"
    exit 1
fi
tag=$2


env_name=lqer
work_dir=$(dirname $(dirname $(dirname $(realpath $0))))
pipeline_run_dir=$work_dir/experiments/pipeline
ckpt_dir=$work_dir/checkpoints/baseline-no-LQER

function run_pipeline() {
    w_precision=$1
    x_precision=$2
    w_block_size=$3
    x_block_size=$4

    # constants
    rank=1
    w_block_size_escape=${w_block_size//,/x}
    x_block_size_escape=${x_block_size//,/x}
    WxAy=W${w_precision}A${x_precision}

    save_dir=$ckpt_dir/$tag/${WxAy}/WB${w_block_size_escape}-XB${x_block_size_escape} && mkdir -p $save_dir

    echo "=========== Pipeline of W${w_precision}A${x_precision} (without Ak Bk) BlockSize=[$w_block_size] ==========="
    cd $pipeline_run_dir
    conda run -n $env_name --no-capture-output python pipeline.py $config_template $tag \
        --project=lqer-baselines-no-LQER \
        --wandb:project=LQER-baselines \
        --checkpoint_path=$save_dir \
        --tags=":ast:['W$w_precision', 'A$x_precision', '${WxAy}', 'w-block-size_[$w_block_size]', 'x-block-size_[$x_block_size]',  'baseline', 'no-LQER']" \
        --approximate:approximator:default:rank=$rank \
        --approximate:approximator:default:W_quantizer:width=$w_precision \
        --approximate:approximator:default:W_quantizer:block_size=:ast:[$w_block_size] \
        --approximate:approximator:default:A_quantizer:width=$x_precision \
        --approximate:approximator:default:A_quantizer:block_size=:ast:[$x_block_size] \
        --approximate:approximator:default:B_quantizer:width=$x_precision \
        --approximate:approximator:default:B_quantizer:block_size=:ast:[$x_block_size] \
        --l_config:linear:rank=$rank \
        --q_config:linear:w_quantizer:width=$w_precision \
        --q_config:linear:w_quantizer:block_size=:ast:[$w_block_size] \
        --q_config:linear:x_quantizer:width=$x_precision \
        --q_config:linear:x_quantizer:block_size=:ast:[$x_block_size] \
        --q_config:linear:b_quantizer:width=$x_precision \
        --q_config:linear:b_quantizer:block_size=:ast:[$x_block_size] \
        --q_config:bmm:w_quantizer:width=$x_precision \
        --q_config:bmm:w_quantizer:block_size=:ast:[$x_block_size] \
        --q_config:bmm:x_quantizer:width=$x_precision \
        --q_config:bmm:x_quantizer:block_size=:ast:[$x_block_size] \
        --evaluate:disable_lqer=:ast:1 \
        --enable_wandb=:ast:0 \
        --enable_profiling=:ast:0 \
        --enable_approximation=:ast:0 \
        --enable_perplexity_evaluation=:ast:1 \
        --enable_harness_downstream_evaluation=:ast:1

    if [ $? -ne 0 ]; then
        echo "❌ Failed to run pipeline of Baseline-no-LQER, rank=$rank, WxAy=$WxAy, W-BlockSize=[$w_block_size], X-BlockSize=[$x_block_size]"
        exit 1
    fi
}

declare -a w_precisions=(4)
declare -a x_precisions=(8)
declare -a w_block_size_options=("1,16")
declare -a x_block_size_options=("1,16")

for wp in "${w_precisions[@]}"; do
    for xp in "${x_precisions[@]}"; do
        for wbs in "${w_block_size_options[@]}"; do
            for xbs in "${x_block_size_options[@]}"; do
                run_pipeline $wp $xp $wbs $xbs
                if [ $? -ne 0 ]; then
                    exit 1
                fi
            done
        done
    done
done
