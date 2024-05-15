import os
import argparse
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling, GenerationConfig
from transformers.integrations import WandbCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datetime import timedelta
from distutils.util import strtobool
from dotenv import load_dotenv
import wandb
from tqdm.auto import tqdm

from common.consts import DS_UPLOAD_PATH, MODELS_DIR
from common.utils import ensure_chat_template, standardize_chat, get_chat
from common.prompts import get_system_prompt

load_dotenv()
WANDB_DIR = ".wandb"
os.environ["WANDB_DIR"] = WANDB_DIR
os.makedirs(WANDB_DIR, exist_ok=True)

models_dir = MODELS_DIR
if os.path.basename(os.getcwd()) != "src":
    models_dir = models_dir[len("../") :]


def get_args():
    parser = argparse.ArgumentParser(prog="finetune", description="Finetune a language model on a dataset of conversations.")
    boolean = lambda x: bool(strtobool(str(x)))
    ######
    parser.add_argument("--dataset_name", type=str, default=DS_UPLOAD_PATH)
    parser.add_argument("--input_model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--output_model_name", type=str, default="rag-tge_Mistral_LoRA")
    parser.add_argument("--resume_from_checkpoint", type=boolean, default=False)
    parser.add_argument("--use_unsloth", type=boolean, default=False)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--system_prompt_id", type=int, default=4)
    ######
    parser.add_argument("--load_in_4bit", type=boolean, default=False, help="QLoRA")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--use_flash_attention_2", type=boolean, default=True)
    ######
    # lora; in paper they recommended 64, 16, 0.1; Sebastian Raschka benchmarks show maybe 256, 128, 0.05 is better
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    ######
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8, help="most performant when kept to multiples of 8")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
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
    parser.add_argument("--data_collator", type=str, default="standard", choices=["standard", "completion"])
    ######
    parser.add_argument("--save_steps_denom", type=int, default=1, help="will set save_steps to 1/n")
    parser.add_argument("--eval_steps_denom", type=int, default=4, help="will set eval_steps to 1/n")
    parser.add_argument("--eval_generation_samples", type=int, default=5)
    parser.add_argument("--push_to_hub", type=boolean, default=True)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
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

    if args.lora_dropout != 0:
        print("WARNING: LoRA dropout is not supported in unsloth, setting it to 0")
    lora_dropout = 0
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        max_seq_length=args.max_seq_length,
        use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
        random_state=args.seed,
        modules_to_save=["lm_head", "embed_tokens"] if not tokenizer.chat_template else None,
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.input_model_name,
        use_fast=False,
    )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"] if not tokenizer.chat_template else None,
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


def format_conversation(args, tokenizer, user_prompt, assistant_prompt):
    messages = [
        {"role": "system", "content": get_system_prompt(args.system_prompt_id)},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prompt},
        {"role": "user", "content": "Thank you!"},
        {"role": "assistant", "content": "You're welcome!"},
    ]
    messages = standardize_chat(args.input_model_name, messages)
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted


def get_formatter(args, tokenizer):
    def format_conversations(rows):
        ## needed if using unsloth, but disabled because it breaks dataset caching
        # if isinstance(rows['prompt'], str):
        #     rows = {'prompt': [rows['prompt']], 'answer': [rows['answer']]}
        #     return format_conversations(rows)[0]
        ##

        formatted = [format_conversation(args, tokenizer, user_prompt=rows["prompt"][i], assistant_prompt=rows["answer"][i]) for i in range(len(rows["prompt"]))]
        max_token_count = max([len(tokenizer.encode(f)) for f in formatted])
        if max_token_count > args.max_seq_length:
            raise ValueError(f"max_token_count: {max_token_count} > args.max_seq_length: {args.max_seq_length}")
        return formatted

    return format_conversations


class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples, args, log_model="checkpoint"):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path)
        self.system_prompt = get_system_prompt(args.system_prompt_id)
        self.model_name = args.output_model_name

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs,
                generation_config=self.gen_config,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                top_k=40,
                top_p=0.9,
            )
        response = outputs[0][inputs.shape[-1] :]
        response = self.tokenizer.decode(response, skip_special_tokens=False)
        return response

    def samples_table(self, global_step):
        records_table = wandb.Table(columns=["global_step", "prompt", "generation"])
        for row in tqdm(self.sample_dataset, leave=False):
            chat = get_chat(self.model_name, user_prompt=row["prompt"], system_prompt=self.system_prompt)
            prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            generation = self.generate(prompt)
            records_table.add_data(global_step, prompt, generation)
        return records_table

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(state.global_step)
        self._wandb.log({"sample_predictions": records_table})


def get_trainer(args, model, tokenizer, peft_config, dataset):
    if args.data_collator == "standard":
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)
    elif args.data_collator == "completion":
        answer_detect_text = "\nQuestion:"
        answer_detect_tokens = tokenizer(answer_detect_text, add_special_tokens=False)["input_ids"]
        original_len = len(answer_detect_tokens)
        # some tokenizers, like mistral, adds some weird tokens (not bos/eos), we need to remove them
        suffixed_tokens = tokenizer(f"{answer_detect_text}...", add_special_tokens=False)["input_ids"]
        answer_detect_tokens = [t for t, s in zip(answer_detect_tokens, suffixed_tokens) if t == s]
        prefixed_tokens = tokenizer(f"...{answer_detect_text}", add_special_tokens=False)["input_ids"]
        answer_detect_tokens = [t for t, s in zip(answer_detect_tokens[::-1], prefixed_tokens[::-1]) if t == s][::-1]
        new_len = len(answer_detect_tokens)
        if original_len != new_len:
            print(f"WARNING: tokenizer reduced the answer_detect_tokens from {original_len} to {new_len}")
        print(f'answer detect tokens: {answer_detect_tokens} => "{tokenizer.decode(answer_detect_tokens)}"')
        data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=answer_detect_tokens)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=get_formatter(args, tokenizer),
        data_collator=data_collator,
        packing=False,
        dataset_num_proc=min(args.dataset_num_proc, os.cpu_count()),  # fails on some systems
        peft_config=peft_config,
        args=TrainingArguments(
            output_dir=f"{models_dir}/{args.output_model_name}",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},  # doesn't hide the warning anyway
            group_by_length=True,
            num_train_epochs=args.epochs,
            evaluation_strategy="steps",
            eval_steps=1 / args.eval_steps_denom,
            save_steps=1 / args.save_steps_denom,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
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
        dataset_kwargs={
            "add_special_tokens": False,  # we are adding it manually in the formatting function
        },
    )

    wandb_callback = LLMSampleCB(trainer=trainer, test_dataset=dataset["test"], num_samples=args.eval_generation_samples, args=args)
    trainer.add_callback(wandb_callback)

    return trainer


def train(args, trainer):
    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_runtime = timedelta(seconds=trainer_stats.metrics["train_runtime"])
    print(f"training took {train_runtime}")


if __name__ == "__main__":
    args = get_args()
    os.environ["WANDB_PROJECT"] = args.output_model_name
    model, tokenizer, peft_config = get_model(args)
    model, tokenizer = ensure_chat_template(model, tokenizer)
    dataset = get_dataset(args)
    trainer = get_trainer(args, model, tokenizer, peft_config, dataset)
    train(args, trainer)
