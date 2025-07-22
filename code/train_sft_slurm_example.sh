#!/bin/bash

#SBATCH --job-name=launch_sft     # Job name
#SBATCH --output=${LOG_DIR}/fs_sft.out # Output file
#SBATCH --partition=spgpu2            # Partition name
#SBATCH --gpus=8                     # Number of GPUs
#SBATCH --mem-per-gpu=55G            # Memory per GPU
#SBATCH --account=${ACCOUNT_NAME}        # Account name (set ACCOUNT_NAME variable below)
#SBATCH --time=10:00:00              # Time limit (10 hours)
#SBATCH --error=${LOG_DIR}/fs_sft.err  # Output file for errors (stderr)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
##SBATCH --nodelist=gl1710

# ===================== USER CONFIGURATION =====================
# Set these variables to your own values
export LOG_DIR="/path/to/your/log/dir"           # <-- Set your log directory
export CONDA_ENV_PATH="/path/to/your/conda/env"  # <-- Set your conda environment path
export CONDA_ENV_NAME="your_conda_env"           # <-- Set your conda environment name
export SAVE_PATH="/path/to/output_dir"           # <-- Set your output directory
export WANDB_API_KEY="your_wandb_api_key"        # <-- Set your wandb API key
# =============================================================

# Load any necessary modules
source ~/.bashrc
cd $LOG_DIR
export PATH=$CONDA_ENV_PATH/bin:$PATH
conda init bash
conda activate $CONDA_ENV_NAME

# --include=localhost:0,1,2,3,5,6
# NOTE: if you are using single node, use this piece of codes; if you are using multi nodes, comment this command and uncomment all the following lines.

deepspeed  --module openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset liusiyang/eeyore_depression_sft \
   --input_key messages \
   --train_batch_size 16 \
   --micro_train_batch_size 2 \
   --apply_chat_template \
   --tokenizer_chat_template "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}" \
   --max_samples 3500 \
   --pretrain meta-llama/Llama-3.1-8B-Instruct \
   --save_path $SAVE_PATH \
   --save_steps -1 \
   --eval_dataset liusiyang/eeyore_depression_sft \
   --eval_split train \
   --logging_steps 1 \
   --eval_steps 30 \
   --zero_stage 3 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --l2 1e-3 \
   --multiturn \
   --learning_rate 5e-6 \
   --use_wandb $WANDB_API_KEY \
   --gradient_checkpointing \


