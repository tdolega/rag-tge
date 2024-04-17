import argparse
import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from datetime import timedelta
from distutils.util import strtobool

from common.consts import DS_UPLOAD_PATH, MODELS_DIR
from common.utils import add_chatml_support

# todo: learn only responses, not prompts (set attention correctly)

WANDB_DIR = ".wandb"
os.environ["WANDB_DIR"] = WANDB_DIR
os.makedirs(WANDB_DIR, exist_ok=True)

models_dir = MODELS_DIR
if os.path.basename(os.getcwd()) != "src":
    models_dir = models_dir[len("../"):]

def get_args():
    parser = argparse.ArgumentParser(prog="finetune", description="Finetune a language model on a dataset of conversations.")
    boolean = lambda x: bool(strtobool(str(x)))
    ######
    parser.add_argument("--dataset_name", type=str, default=DS_UPLOAD_PATH)
    parser.add_argument("--input_model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output_model_name", type=str, default="rag-tge_TinyLlama_LoRA")
    parser.add_argument("--resume_from_checkpoint", type=boolean, default=False)
    parser.add_argument("--use_unsloth", type=boolean, default=False)
    ######
    parser.add_argument("--load_in_4bit", type=boolean, default=False, help="QLoRA")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--use_flash_attention_2", type=boolean, default=True)
    ######
    # lora; in paper they recommended 64, 16, 0.1; Sebastian Raschka benchmarks show maybe 256, 128, 0.05 is better
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    ######
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2, help="most performant when kept to multiples of 8")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="default is 1.0")
    parser.add_argument("--gradient_checkpointing", type=boolean, default=True)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--neftune_noise_alpha", type=float, default=5, help="default is None")
    parser.add_argument("--bf16", type=boolean, default=True)
    parser.add_argument("--tf32", type=boolean, default=True)
    parser.add_argument("--dataset_num_proc", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    ######
    parser.add_argument("--save_steps_denom", type=int, default=4, help="will set save_steps to 1/n")
    parser.add_argument("--eval_steps_denom", type=int, default=32, help="will set eval_steps to 1/n")
    parser.add_argument("--push_to_hub", type=boolean, default=True)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    ######
    parser.add_argument("--seed", type=int, default=50)
    ######
    args = parser.parse_args()
    print(">>> args:\n", "\n".join([f"{k}: {v}" for k, v in vars(args).items()]), "\n<<<")
    return args


def get_model_unsloth(args):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.input_model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # autodetect
        load_in_4bit=args.load_in_4bit,
        use_cache=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        max_seq_length=args.max_seq_length,
        use_gradient_checkpointing=args.gradient_checkpointing,
        random_state=args.seed,
        modules_to_save=["lm_head", "embed_tokens"],  # tokenizer changes
    )

    peft_config = None

    return model, tokenizer, peft_config


def get_model_transformers(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        if args.load_in_4bit
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.input_model_name,
        device_map={"": 0},  # one gpu for now
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 else None,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.input_model_name, use_fast=False)

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],  # tokenizer changes
    )

    return model, tokenizer, peft_config


def get_model(args):
    if args.use_unsloth:
        return get_model_unsloth(args)
    else:
        return get_model_transformers(args)


def get_dataset(args):
    dataset = load_dataset(args.dataset_name)
    print(dataset)
    if args.limit_train:
        dataset["train"] = dataset["train"].select(range(args.limit_train))
    if args.limit_test:
        dataset["test"] = dataset["test"].select(range(args.limit_test))
    if args.limit_train or args.limit_test:
        print("limited dataset:")
        print(dataset)
    return dataset


def format_conversation(tokenizer, system_prompt, user_prompt, assistant_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False).strip()


def get_formatter(tokenizer):
    def format_conversations(rows):
        return [format_conversation(tokenizer, rows["system_prompt"][i], rows["user_prompt"][i], rows["answer"][i]) for i in range(len(rows["system_prompt"]))]

    return format_conversations


def get_trainer(args, model, tokenizer, peft_config, dataset):
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=get_formatter(tokenizer),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8),
        packing=False,
        dataset_num_proc=min(args.dataset_num_proc, os.cpu_count()),  # fails on some systems
        dataset_kwargs={
            # "add_special_tokens": False, # we are adding it manually in the formatting function
        },
        peft_config=peft_config,
        args=TrainingArguments(
            output_dir=f"{models_dir}/{args.output_model_name}",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            group_by_length=True,
            num_train_epochs=args.epochs,
            evaluation_strategy="steps",
            eval_steps=1 / args.eval_steps_denom,
            save_steps=1 / args.save_steps_denom,
            save_total_limit=args.save_total_limit,
            push_to_hub=args.push_to_hub,
            hub_private_repo=True,
            hub_strategy="checkpoint",
            report_to="wandb",
            logging_steps=1,
            optim=args.optim,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            neftune_noise_alpha=args.neftune_noise_alpha,
            bf16=args.bf16,
            tf32=args.tf32,
            seed=args.seed,
            dataloader_num_workers=min(args.dataloader_num_workers, args.batch_size, os.cpu_count()),
            # torch_compile=True, # crashes
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


def train(args, trainer):
    print_memory()
    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print_memory()
    train_runtime = timedelta(seconds=trainer_stats.metrics["train_runtime"])
    print(f"training took {train_runtime}")


if __name__ == "__main__":
    args = get_args()
    os.environ["WANDB_PROJECT"] = args.output_model_name
    model, tokenizer, peft_config = get_model(args)
    model, tokenizer = add_chatml_support(model, tokenizer)
    dataset = get_dataset(args)
    trainer = get_trainer(args, model, tokenizer, peft_config, dataset)
    train(args, trainer)
