#!/bin/bash

# Lista de modelos
models=("EleutherAI/pythia-14m" "EleutherAI/pythia-70m-deduped")

# Parámetros comunes
epochs=5
noise_alpha=5
wandb=true

# Iterar sobre cada modelo en la lista
for model_name in "${models[@]}"; do
  # Construir la lista de combinaciones de parámetros
  params=(
    "--model_name $model_name --epochs $epochs --wandb $wandb"
    "--model_name $model_name --epochs $epochs --wandb $wandb --neftune_noise_alpha $noise_alpha"
    "--model_name $model_name --epochs $epochs --wandb $wandb --instruction_modelling true"
    "--model_name $model_name --epochs $epochs --wandb $wandb --neftune_noise_alpha $noise_alpha --instruction_modelling true"
  )

  # Construir el comando final ejecutando python train.py con todas las combinaciones de parámetros
  for param in "${params[@]}"; do
    python train.py $param
  done
done
