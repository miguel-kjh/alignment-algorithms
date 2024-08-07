#!/bin/bash

models=("EleutherAI/pythia-14m" "EleutherAI/pythia-70m-deduped" "EleutherAI/pythia-410m-deduped" "EleutherAI/pythia-1b-deduped" "microsoft/phi-2")

epochs=5
noise_alpha=5
wandb=true
default_batch_size=8
default_block_size=512

for model_name in "${models[@]}"; do
  if [[ "$model_name" == "EleutherAI/pythia-1b-deduped" ]]; then
    batch_size=4
    block_size=256
    extra_params="--qlora true"
    epochs=2
  elif [[ "$model_name" == "microsoft/phi-2" ]]; then
    batch_size=4
    block_size=256
    extra_params="--qlora true --lora_target_modules "Wqkv,fc1,fc2" --lora_r 8 --lora_alpha 16"
    epochs=2
  else
    batch_size=$default_batch_size
    block_size=$default_block_size
    extra_params=""
  fi

  params=(
    "--model_name $model_name --epochs $epochs --idda GAIR/lima --upload true --wandb $wandb --batch_size $batch_size --block_size $block_size $extra_params"
    #"--model_name $model_name --epochs $epochs --idda GAIR/lima --upload true --wandb $wandb --neftune_noise_alpha $noise_alpha --batch_size $batch_size --block_size $block_size $extra_params"
    #"--model_name $model_name --epochs $epochs --upload true --wandb $wandb --instruction_modelling true --batch_size $batch_size --block_size $block_size $extra_params"
    #"--model_name $model_name --epochs $epochs --upload true --wandb $wandb --neftune_noise_alpha $noise_alpha --instruction_modelling true --batch_size $batch_size --block_size $block_size $extra_params"
  )

  echo "Model: $model_name"

  for param in "${params[@]}"; do
    python train.py $param
  done
done

