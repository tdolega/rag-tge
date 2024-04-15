import argparse
import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from datetime import timedelta

from common.consts import DS_UPLOAD_PATH
from common.utils import add_chatml_support

# todo: make this into functions
# todo: move all those constants to args
# todo: learn only responses, not prompts (set attention correctly)

# general
dataset_name = DS_UPLOAD_PATH
input_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_model_name = "rag-tge_TinyLlama_LoRA"
resume_from_checkpoint = False
use_unsloth = False

# base
load_in_4bit = False
max_seq_length = 2048
use_flash_attention_2 = True

# lora; in paper they recommended 64, 16, 0.1; Sebastian Raschka benchmarks show maybe 256, 128, 0.05 is better
lora_rank = 128
lora_alpha = 64
lora_dropout = 0.1

# dataset
# dataloader_num_workers = 4
limit_train = None
limit_test = None

# training
batch_size = 2  # most performant when kept to multiples of 8
gradient_accumulation_steps = 4
epochs = 2
warmup_ratio = 0.05
weight_decay = 0.01
max_grad_norm = 0.3  # default is 1.0
gradient_checkpointing = True
learning_rate = 2e-4
lr_scheduler_type = "linear"
optim = "paged_adamw_32bit"  # ['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_torch_npu_fused', 'adamw_apex_fused', 'adafactor', 'adamw_anyprecision', 'sgd', 'adagrad', 'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit', 'lion_32bit', 'paged_adamw_32bit', 'paged_adamw_8bit', 'paged_lion_32bit', 'paged_lion_8bit', 'rmsprop', 'rmsprop_bnb', 'rmsprop_bnb_8bit', 'rmsprop_bnb_32bit']
neftune_noise_alpha = 5  # default is None
mixed_precision_training = True
tf32 = True

# checkpointing
save_steps = 1 / 8
eval_steps = 1 / 8
push_to_hub = True
save_total_limit = 1

# other
seed = 50

##########

os.environ["WANDB_PROJECT"] = output_model_name
# os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
os.makedirs(".wandb", exist_ok=True)

##########

print("bf16:", torch.cuda.is_bf16_supported())
print("cuda:", torch.version.cuda)

##########

if use_unsloth:
    #############
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=input_model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # autodetect
        load_in_4bit=load_in_4bit,
        use_cache=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        max_seq_length=max_seq_length,
        use_gradient_checkpointing=gradient_checkpointing,
        random_state=seed,
        modules_to_save=["lm_head", "embed_tokens"],  # tokenizer changes
    )

    peft_config = None
#############
else:
    #############
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        if load_in_4bit
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        input_model_name,
        device_map={"": 0},  # one gpu for now
        attn_implementation="flash_attention_2" if use_flash_attention_2 else None,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(input_model_name, use_fast=False)

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],  # tokenizer changes
    )
#############

# tokenizer.padding_side = "left" # controversial
if tokenizer.pad_token == None:
    print("setting pad token to eos token")
    tokenizer.pad_token = tokenizer.eos_token

##########

print(f"old bos token: {tokenizer.bos_token}, old eos token: {tokenizer.eos_token}, old pad token: {tokenizer.pad_token}")

model, tokenizer = add_chatml_support(model, tokenizer)

##########

from datasets import load_dataset

dataset = load_dataset(dataset_name)
print(dataset)
if limit_train:
    dataset["train"] = dataset["train"].select(range(limit_train))
if limit_test:
    dataset["test"] = dataset["test"].select(range(limit_test))
print(dataset)


def format_conversation(system_prompt, user_prompt, assistant_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prompt},
    ]
    templated = tokenizer.apply_chat_template(messages, tokenize=False).strip()
    return templated


def format_conversations(rows):
    return [format_conversation(rows["system_prompt"][i], rows["user_prompt"][i], rows["answer"][i]) for i in range(len(rows["system_prompt"]))]


##########

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    formatting_func=format_conversations,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8),
    packing=False,
    dataset_num_proc=min(8, os.cpu_count()),  # fails on some systems
    dataset_kwargs={
        # "add_special_tokens": False, # we are adding it manually in the formatting function
    },
    peft_config=peft_config,
    args=TrainingArguments(
        output_dir=output_model_name,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        group_by_length=True,
        num_train_epochs=epochs,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        push_to_hub=push_to_hub,
        hub_private_repo=True,
        hub_strategy="checkpoint",
        report_to="wandb",
        logging_steps=1,
        optim=optim,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        neftune_noise_alpha=neftune_noise_alpha,
        bf16=mixed_precision_training,
        tf32=tf32,
        seed=seed,
        # dataloader_num_workers=min(dataloader_num_workers, batch_size, os.cpu_count()),
        # include_tokens_per_second=True,
        # torch_compile=True, # crashes with error
    ),
)


def print_memory():
    all_memory_used = 0
    all_memory_total = 0
    for i in range(torch.cuda.device_count()):
        gpu_stats = torch.cuda.get_device_properties(i)
        memory_used = torch.cuda.max_memory_reserved(i) / 1024 / 1024 / 1024
        memory_total = gpu_stats.total_memory / 1024 / 1024 / 1024
        all_memory_used += memory_used
        all_memory_total += memory_total
        print(f"GPU {i}: {gpu_stats.name} (used {memory_used:.1f} of {memory_total:.1f} GiB)")
    if torch.cuda.device_count() > 1:
        print(f"total {all_memory_used:.1f} of {all_memory_total:.1f} GiB of memory reserved")


print_memory()

trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

print_memory()
train_runtime = timedelta(seconds=trainer_stats.metrics["train_runtime"])
print(f"training took {train_runtime}")

##########
