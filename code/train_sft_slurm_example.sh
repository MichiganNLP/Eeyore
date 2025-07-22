#!/bin/bash

#SBATCH --job-name=launch_sft     # Job name
#SBATCH --output=/nfs/turbo/coe-mihalcea/lsiyang/Eeyore/log/fs_dpo.out # Output file
#SBATCH --partition=spgpu2            # Partition name
#SBATCH --gpus=8                     # Number of GPUs
#SBATCH --mem-per-gpu=55G            # Memory per GPU
#SBATCH --account=mihalcea_owned1        # Account name
#SBATCH --time=10:00:00              # Time limit (8 hours)
#SBATCH --error=/nfs/turbo/coe-mihalcea/lsiyang/Eeyore/log/fs_dpo.err  # Output file for errors (stderr)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
##SBATCH --nodelist=gl1710


# Load any necessary modules
source ~/.bashrc
cd ~/Eeyore/log
export PATH=~/miniconda/envs/llmdepression/bin:$PATH
conda init bash
conda activate llmdepression




# --include=localhost:0,1,2,3,5,6
# NOTE: if you are using single node, use this piece of codes; if you are using multi nodes, comment this ccommand and uncomment all the following lines.

deepspeed  --module openrlhf.cli.train_dpo \
   --save_path ../output_dir/eeyore_sft_epoch2_dpo_round1_epoch1_llama3.1_8B \
   --save_steps 30 \
   --logging_steps 30 \
   --eval_dataset liusiyang/eeyore_depression_generated_preference \
   --eval_steps 200 \
   --eval_split train \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --pretrain ../output_dir/eeyore_sft_llama3.1_8B_epoch2 \
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
   --use_wandb 088049257f73ae15cc8913b6a397e430af2af571 \
   --ref_offload 
