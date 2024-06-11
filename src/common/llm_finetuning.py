import os
import argparse
import torch
from datasets import load_dataset, load_from_disk
from transformers import TrainingArguments, DataCollatorForLanguageModeling, GenerationConfig
from transformers.integrations import WandbCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datetime import timedelta
from distutils.util import strtobool
from dotenv import load_dotenv
import wandb
from tqdm.auto import tqdm

load_dotenv()

boolean = lambda x: bool(strtobool(str(x)))

WANDB_DIR = ".wandb"
os.environ["WANDB_DIR"] = WANDB_DIR
os.makedirs(WANDB_DIR, exist_ok=True)


def ensure_chat_template(model, tokenizer):
    # tokenizer.padding_side = "left" # controversial
    if tokenizer.pad_token == None:
        if "llama-3" in model.name_or_path.lower():
            print("setting pad token to '<|end_of_text|>'")  # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/101
            tokenizer.pad_token = "<|end_of_text|>"
        elif tokenizer.unk_token != None:
            assert tokenizer.unk_token != tokenizer.eos_token, "unk token is eos token"
            print("setting pad token to unk token")
            tokenizer.pad_token = tokenizer.unk_token
        else:
            print("WARNING: setting pad token to eos token")
            tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is not None:
        print("chat template already set")
        return model, tokenizer

    PAD_TOKEN = "</s>"
    BOS_TOKEN = "<|im_start|>"
    EOS_TOKEN = "<|im_end|>"

    print(f"old bos token: {tokenizer.bos_token}, old eos token: {tokenizer.eos_token}, old pad token: {tokenizer.pad_token}")

    # tokenizer.pad_token = PAD_TOKEN
    tokenizer.add_tokens([BOS_TOKEN])
    tokenizer.add_special_tokens(dict(eos_token=EOS_TOKEN))

    # ChatML template, from https://huggingface.co/docs/transformers/main/chat_templating
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer


user_notified_about_standardize_chat = False


def standardize_chat(model_name, chat):
    if chat[0]["role"] != "system":
        return chat
    LLMS_WITHOUT_SYSTEM_PROMPT = [
        "mixtral",
        "mistral",
        "gemma",
    ]
    if not any([name in model_name.lower() for name in LLMS_WITHOUT_SYSTEM_PROMPT]):
        return chat

    raise NotImplementedError("standardize_chat is not implemented for LLMSampleCB")

    global user_notified_about_standardize_chat
    if not user_notified_about_standardize_chat:
        print("WARNING: standardize_chat is enabled for this model")
        user_notified_about_standardize_chat = True
    pseudo_system = [
        {"role": "user", "content": chat[0]["content"]},
        {"role": "assistant", "content": "Ok."},
    ]
    rest = chat[1:]
    return pseudo_system + rest


class LLMSampleCB(WandbCallback):
    def __init__(self, args, trainer, test_dataset, row_to_messages):
        super().__init__()
        self._log_model = False
        self.sampled_dataset = test_dataset.select(range(args.eval_generation_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.generation_config = GenerationConfig.from_pretrained(trainer.model.name_or_path)
        self.model_name = args.output_model_name
        self.row_to_messages = row_to_messages

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs,
                generation_config=self.generation_config,
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
        for row in tqdm(self.sampled_dataset, leave=False):
            chat = self.row_to_messages(row)

            if chat[0]["role"] == "system":
                chat = chat[:2]
            else:
                chat = chat[:1]

            prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            response = self.generate(prompt)
            records_table.add_data(global_step, prompt, response)
        return records_table

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(state.global_step)
        self._wandb.log({"sample_predictions": records_table})


class LLM_TRAINER_ASSISTANT:
    def add_args(self, parser):
        ######
        parser.add_argument("--dataset_name", type=str, default=None)
        parser.add_argument("--input_model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
        parser.add_argument("--output_model_name", type=str, required=True)
        parser.add_argument("--output_dir", type=str, default=".")
        parser.add_argument("--resume_from_checkpoint", type=boolean, default=False)
        parser.add_argument("--use_unsloth", type=boolean, default=False)
        parser.add_argument("--seed", type=int, default=50)
        ######
        parser.add_argument("--load_in_4bit", type=boolean, default=False, help="QLoRA")
        parser.add_argument("--max_seq_length", type=int, default=8192)
        parser.add_argument("--use_flash_attention_2", type=boolean, default=True)
        ######
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
        parser.add_argument("--completion_collator_needle", type=str, default=None)
        ######
        parser.add_argument("--save_steps_denom", type=int, default=1, help="will set save_steps to 1/n")
        parser.add_argument("--eval_steps_denom", type=int, default=4, help="will set eval_steps to 1/n")
        parser.add_argument("--eval_generation_samples", type=int, default=5)
        parser.add_argument("--push_to_hub", type=boolean, default=False)
        parser.add_argument("--save_total_limit", type=int, default=1)
        parser.add_argument("--limit_train", type=int, default=None)
        parser.add_argument("--limit_test", type=int, default=None)
        ######
        return parser

    def __init__(self, parser=None, force_args={}):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser = self.add_args(parser)
        self.args = parser.parse_args()
        for arg_name, arg_value in force_args.items():
            assert hasattr(self.args, arg_name), f"unsupported default arg: {arg_name}"
            setattr(self.args, arg_name, arg_value)
        print("> args:")
        print("\n".join([f"  {k}: {v}" for k, v in vars(self.args).items()]))

        os.environ["WANDB_PROJECT"] = self.args.output_model_name

        if self.args.use_unsloth:
            self.init_model_unsloth()
        else:
            self.init_model_transformers()
        self.init_data_collator()
        self.init_dataset()
        self.row_to_messages = None

    def init_model_unsloth(self):
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.args.input_model_name,
            max_seq_length=self.args.max_seq_length,
            dtype=None,  # autodetect
            load_in_4bit=self.args.load_in_4bit,
            use_cache=False,
        )

        if self.args.lora_dropout != 0:
            print("> WARNING: LoRA dropout is not supported in unsloth, setting it to 0")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=0,
            bias="none",
            max_seq_length=self.args.max_seq_length,
            use_gradient_checkpointing="unsloth" if self.args.gradient_checkpointing else False,
            random_state=self.args.seed,
            modules_to_save=["lm_head", "embed_tokens"] if not self.tokenizer.chat_template else None,
        )

        self.peft_config = None

    def init_model_transformers(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=self.args.load_in_4bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
            if self.args.load_in_4bit
            else None
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.input_model_name,
            device_map={"": 0},  # one gpu for now
            attn_implementation="flash_attention_2" if self.args.use_flash_attention_2 else None,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.input_model_name,
            use_fast=False,
        )

        self.peft_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_tokens"] if not self.tokenizer.chat_template else None,
        )

    def init_dataset(self):
        assert self.args.dataset_name, "dataset_name must be provided"
        if self.args.dataset_name.startswith("."):
            ds = load_from_disk(self.args.dataset_name)
        else:
            ds = load_dataset(self.args.dataset_name)
        print(ds)
        if self.args.limit_train:
            ds["train"] = ds["train"].select(range(self.args.limit_train))
        if self.args.limit_test:
            ds["test"] = ds["test"].select(range(self.args.limit_test))
        if self.args.limit_train or self.args.limit_test:
            print("> limited dataset:")
            print(ds)
        self.dataset = ds

    def dataset_length(self):
        return sum(len(self.dataset[split]) for split in ["train", "test"])

    def format_conversation(self, row):
        messages = self.row_to_messages(row)
        messages = standardize_chat(self.args.input_model_name, messages)
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return formatted

    def get_formatter(self):
        def format_conversations(rows):
            keys = [k for k in rows]
            probably_is_batch = all(type(rows[k]) == list for k in keys)
            if probably_is_batch:
                return [self.format_conversation({k: v for k, v in zip(keys, values)}) for values in zip(*[rows[k] for k in keys])]
            return self.format_conversation(rows)

        return format_conversations

    def remove_overflowing_samples(self):
        formatter = self.get_formatter()

        def fits_in_context(rows):
            formatted = formatter(rows)
            return [len(self.tokenizer(f, return_tensors="pt")["input_ids"][0]) <= self.args.max_seq_length for f in formatted]

        len_before = self.dataset_length()
        self.dataset = self.dataset.filter(fits_in_context, batched=True)
        len_after = self.dataset_length()
        print(f"> WARNING: removed {len_before - len_after} samples that were too long")
        print(self.dataset)

    def init_data_collator(self):
        match self.args.data_collator:
            case "standard":
                self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False, pad_to_multiple_of=8)
            case "completion":
                assert self.args.completion_collator_needle is not None, "completion_collator_needle must be provided for completion data_collator"
                answer_detect_tokens = self.tokenizer(self.args.completion_collator_needle, add_special_tokens=False)["input_ids"]
                original_len = len(answer_detect_tokens)
                # some tokenizers, like mistral, adds some weird tokens (not bos/eos), we need to remove them
                suffixed_tokens = self.tokenizer(f"{self.args.completion_collator_needle}...", add_special_tokens=False)["input_ids"]
                answer_detect_tokens = [t for t, s in zip(answer_detect_tokens, suffixed_tokens) if t == s]
                prefixed_tokens = self.tokenizer(f"...{self.args.completion_collator_needle}", add_special_tokens=False)["input_ids"]
                answer_detect_tokens = [t for t, s in zip(answer_detect_tokens[::-1], prefixed_tokens[::-1]) if t == s][::-1]
                new_len = len(answer_detect_tokens)
                if original_len != new_len:
                    print(f"> WARNING: tokenizer reduced the answer_detect_tokens from {original_len} to {new_len}")
                print(f'> answer detect tokens: {answer_detect_tokens} => "{self.tokenizer.decode(answer_detect_tokens)}"')
                self.data_collator = DataCollatorForCompletionOnlyLM(tokenizer=self.tokenizer, response_template=answer_detect_tokens)
            case _:
                raise ValueError(f"unsupported data_collator: {self.args.data_collator}")

    def init_trainer(self):
        args = self.args
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            max_seq_length=args.max_seq_length,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            formatting_func=self.get_formatter(),
            data_collator=self.data_collator,
            packing=False,
            dataset_num_proc=min(args.dataset_num_proc, os.cpu_count()),  # fails on some systems
            peft_config=self.peft_config,
            args=TrainingArguments(
                output_dir=f"{args.output_dir}/{args.output_model_name}",
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                gradient_checkpointing=args.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},  # doesn't hide the warning anyway
                group_by_length=True,
                num_train_epochs=args.epochs,
                eval_strategy="steps",
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

        self.trainer.add_callback(
            LLMSampleCB(
                args=args,
                trainer=self.trainer,
                test_dataset=self.dataset["test"],
                row_to_messages=self.row_to_messages,
            )
        )

    def prepare(self):
        assert self.row_to_messages, "row_to_messages must be set"
        self.model, self.tokenizer = ensure_chat_template(self.model, self.tokenizer)
        self.remove_overflowing_samples()
        self.init_trainer()

    def train(self):
        trainer_stats = self.trainer.train(resume_from_checkpoint=self.args.resume_from_checkpoint)
        train_runtime = timedelta(seconds=trainer_stats.metrics["train_runtime"])
        print(f"> training took {train_runtime}")

    def __call__(self):
        self.prepare()
        self.train()
