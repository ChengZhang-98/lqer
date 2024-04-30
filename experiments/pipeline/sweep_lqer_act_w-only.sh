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
ckpt_dir=$work_dir/checkpoints/LQER-act

function run_pipeline() {
    rank=$1
    w_precision=$2
    w_block_size=$4

    w_block_size_escape=${w_block_size//,/x}
    WxAy=W${w_precision}

    # constants
    num_profile_samples=64

    # besides modifying toml template, you can override downstream tasks by adding the following line to pipeline.py
    # --evaluate:harness_downstream:datasets=":ast:['arc_easy', 'lambada_openai', 'piqa', 'arc_challenge', 'boolq', 'openbookqa']" \

    save_dir=$ckpt_dir/$tag/${WxAy}/WB${w_block_size_escape}/rank${rank} && mkdir -p $save_dir

    echo "=========== Pipeline of LQER-act-w-only, rank=$rank, WxAy=${WxAy}, W-BlockSize=[$w_block_size] ==========="
    cd $pipeline_run_dir
    conda run -n $env_name --no-capture-output python pipeline.py $config_template $tag \
        --project=lqer-act-sweep \
        --wandb:project=LQER-act \
        --checkpoint_path=$save_dir \
        --tags=":ast:['W$w_precision', 'A$x_precision', '$WxAy', 'w-block-size_[$w_block_size]', 'profile-cnt_${num_profile_samples}', 'LQER-act']" \
        --profile:num_samples=$num_profile_samples \
        --approximate:name="lqer-act" \
        --approximate:approximator:default:rank=$rank \
        --approximate:approximator:default:W_quantizer:width=$w_precision \
        --approximate:approximator:default:W_quantizer:block_size=:ast:[$w_block_size] \
        --approximate:approximator:default:A_quantizer:name="passthrough" \
        --approximate:approximator:default:B_quantizer:name="passthrough" \
        --l_config:linear:rank=$rank \
        --q_config:linear:w_quantizer:width=$w_precision \
        --q_config:linear:w_quantizer:block_size=:ast:[$w_block_size] \
        --q_config:linear:x_quantizer:name="passthrough" \
        --q_config:linear:b_quantizer:name="passthrough" \
        --q_config:bmm:w_quantizer:name="passthrough" \
        --q_config:bmm:x_quantizer:name="passthrough" \
        --q_config:matmul:w_quantizer:name="passthrough" \
        --q_config:matmul:x_quantizer:name="passthrough" \
        --enable_wandb=:ast:0 \
        --enable_profiling=:ast:1 \
        --enable_approximation=:ast:1 \
        --enable_perplexity_evaluation=:ast:1 \
        --enable_harness_downstream_evaluation=:ast:1

    # 📝 check these --enable_xxx

    if [ $? -ne 0 ]; then
        echo "❌ Failed to run pipeline of LQER-act (w-only), rank=$rank, WxAy=$WxAy, W-BlockSize=[$w_block_size]"
        exit 1
    fi
}

declare -a rank_options=(32)
declare -a w_precisions=(4)
declare -a w_block_size_options=("1,-1")

for r in "${rank_options[@]}"; do
    for w_p in "${w_precisions[@]}"; do
        for w_bs in "${w_block_size_options[@]}"; do
            run_pipeline $r $w_p $x_p $w_bs $x_bs
            if [ $? -ne 0 ]; then
                exit 1
            fi
        done
    done
done
