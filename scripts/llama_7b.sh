#!/bin/bash

# Set common variables
model="decapoda-research/llama-7b-hf"
sparsity_ratio=0.5
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define calib_samples values
calib_samples_values=(1 4 16 64 128) 
seed_values=(0 1 2)

# Loop through calib_samples values
for calib_samples in "${calib_samples_values[@]}"; do
    for seed in "${seed_values[@]}"; do
    
        echo "Running with calib_samples=$calib_samples and seed=$seed"

        # Define function to run python command
        run_python_command () {
            python main.py \
            --model $model \
            --seed $seed \
            --nsamples $calib_samples \
            --prune_method $1 \
            --sparsity_ratio $sparsity_ratio \
            --sparsity_type $2 \
            --save "out/${model_name}/$2/$1/${calib_samples}_samples/${seed}_seed"
        }

        # llama-7b with wanda pruning method
        echo "Running with wanda pruning method"
        run_python_command "wanda" "2:4" 
        run_python_command "wanda" "4:8"
        echo "Finished wanda pruning method"

        # llama-7b with wanda_connect pruning method
        echo "Running with connect pruning method"
        run_python_command "wanda_connect" "2:4" 
        run_python_command "wanda_connect" "4:8"
        echo "Finished connect pruning method"
    
    done
done

echo "All experiments completed."