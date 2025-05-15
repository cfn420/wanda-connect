#!/bin/bash
read -p "Enter start layer: " START
read -p "Enter end layer: " END
read -p "Enter device: " DEVICE

eval LAYER=({$START..$END})
CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama2_7b/unstructured/wanda/ \
    --modify_layer ${LAYER[@]} --modify_type o down \
    --eval_zero_shot
