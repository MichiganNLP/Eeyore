<h1 align="center">  <img src="https://github.com/MichiganNLP/Eeyore/blob/main/icon.png" width="50" height="50"> Eeyore: Realistic Depression Simulation  <br /> with Expert-in-the-Loop Supervised and Preference Optimization </h1>

# Will Update Soon

* Model Checkpoints
* Preference Generation Codes
* Reproducing Codes of Baselines

  Email me if you need the above materials earlier


# Data
|Stage|Link|
|-------|-------|
|Seperated Profiles and Dialogues|[liusiyang/eeyore_profile](https://huggingface.co/datasets/liusiyang/eeyore_profile)|
|Instruction-tuning Data for SFT|[liusiyang/eeyore_depression_sft](https://huggingface.co/datasets/liusiyang/eeyore_depression_sft)|
|Model-generated Preference Data for DPO|[liusiyang/eeyore_depression_generated_preference](https://huggingface.co/datasets/liusiyang/eeyore_depression_generated_preference)|


# Model Training

[SFT Script with Slurm](https://github.com/MichiganNLP/Eeyore/blob/main/code/train_sft_slurm_example.sh)
```
# Load any necessary modules
source ~/.bashrc
cd $LOG_DIR
export PATH=$CONDA_ENV_PATH/bin:$PATH
conda init bash
conda activate $CONDA_ENV_NAME
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
```

[DPO Script with Slurm](https://github.com/MichiganNLP/Eeyore/blob/main/code/train_dpo_slurm_example.sh)

```
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
```

# Model Deployment

We provide an OpenAI similar web server for easy integration and evaluation. The server accepts standard OpenAI chat completion requests and returns responses in the same format.

## Start with Generation Parameters used in Experiments
```bash
export PYTHONPATH=${project_dir}
# cd where you put the eeyore model
cd ./output_dir
python ../code/deploy_eeyore.py \
  --model ${model_name} \
  --host 127.0.0.1 \
  --port 6416 \
  --temperature 1.0 \
  --top-p 0.8 \
  --max-new-tokens 4096 \
  --sequence-bias "[[[128009], -4.0]]" \
  --exponential-decay-length-penalty 0 1.01
```

## API Usage
The server provides OpenAI-compatible endpoints:

### Chat Completions
```bash
curl -X POST http://127.0.0.1:6416/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "How are you feeling today?"}
    ]
  }'
```


# Automatic Evaluation

### Reproducing Eeyore and Evaluating

First, deploy the Eeyore model as mentioned in Model Deployment.

```bash
export PYTHONPATH=${project_dir}
python ./code/automatic_eval.py --experiment-name=eeyore --experiment-model=${model_name} --base-url="http://127.0.0.1:6416/v1"
```

### Reproducing Baselines and Evaluating

```
export PYTHONPATH=${project_dir}
python ./code/automatic_eval.py --experiment-name=roleplay-doh --experiment-model="gpt-4o-2024-08-06" --base-url="https://api.openai.com/v1" --api-key=${api_key} --temperature=0.7 --top-p=1.0

```

```
export PYTHONPATH=${project_dir}
python ./code/automatic_eval.py --experiment-name=patient-psi --experiment-model="gpt-4o-2024-08-06" --base-url="https://api.openai.com/v1" --api-key=${api_key} --temperature=0.8 --top-p=1.0

```



