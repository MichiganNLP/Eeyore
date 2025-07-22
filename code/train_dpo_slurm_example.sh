#!/bin/bash

#SBATCH --job-name=launch_dpo     # Job name
#SBATCH --output=${LOG_DIR}/fs_dpo.out # Output file
#SBATCH --partition=spgpu2            # Partition name
#SBATCH --gpus=8                     # Number of GPUs
#SBATCH --mem-per-gpu=55G            # Memory per GPU
#SBATCH --account=${ACCOUNT_NAME}        # Account name (set ACCOUNT_NAME variable below)
#SBATCH --time=10:00:00              # Time limit (10 hours)
#SBATCH --error=${LOG_DIR}/fs_dpo.err  # Output file for errors (stderr)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
##SBATCH --nodelist=gl1710

# ===================== USER CONFIGURATION =====================
# Set these variables to your own values
export LOG_DIR="/path/to/your/log/dir"           # <-- Set your log directory
export CONDA_ENV_PATH="/path/to/your/conda/env"  # <-- Set your conda environment path
export CONDA_ENV_NAME="your_conda_env"           # <-- Set your conda environment name
export SFT_MODEL="/path/to/your/pretrained/model" # <-- Set your pretrained model path
export SAVE_PATH="/path/to/output_dir"           # <-- Set your output directory
export WANDB_API_KEY="your_wandb_api_key"        # <-- Set your wandb API key
# =============================================================

# Load any necessary modules
source ~/.bashrc
cd $LOG_DIR
export PATH=$CONDA_ENV_PATH/bin:$PATH
conda init bash
conda activate $CONDA_ENV_NAME


deepspeed  --module openrlhf.cli.train_dpo \
   --save_path $SAVE_PATH \
   --save_steps 30 \
   --logging_steps 30 \
   --eval_dataset liusiyang/eeyore_depression_generated_preference \ #Just to see the metrics on the training set
   --eval_steps 200 \
   --eval_split train \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --pretrain $SFT_MODEL \
   --bf16 \
   --max_epochs 1 \
   --max_len 5120 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --dataset liusiyang/eeyore_depression_generated_preference \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --prompt_key prompt \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --ref_offload \



   