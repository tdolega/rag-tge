import os
import re
from datasets import load_dataset, concatenate_datasets
import random
import hashlib

from common.prompts import PROMPTS
from common.consts import EVAL_SIZE


def get_start_idx(filename):
    if not os.path.exists(filename):
        return 0

    start_idx = 0
    with open(filename, "r") as output_handle:
        for line in output_handle:
            start_idx += 1
    print(f"resuming from index {start_idx}")
    return start_idx


def get_dataset(limit=None, split="train", level="easy", seed_string="none"):
    dataset = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    if level is not None:
        dataset = dataset.filter(lambda row: row["level"] == level, num_proc=os.cpu_count())
    dataset = dataset.shuffle(seed=102)
    if limit is not None:
        dataset = dataset.select(range(limit))

    if limit <= EVAL_SIZE:
        return dataset

    eval_split = dataset.select(range(EVAL_SIZE))
    train_split = dataset.select(range(EVAL_SIZE, len(dataset)))
    # generate different train data with each model
    int_seed = int(hashlib.md5(seed_string.encode()).hexdigest(), 16) % 2**32
    train_split = train_split.shuffle(seed=int_seed)
    merged_dataset = concatenate_datasets([eval_split, train_split])
    return merged_dataset


def remove_citations(sentence):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sentence)).replace(" |", "").replace("]", "").strip()


def merge_sentences(sentences):
    return " ".join(sentences).replace("\n", " ").strip()


def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = remove_citations(sentence)
    return re.sub(r"[^a-z0-9 ]", "", sentence)  # keep only alphanumeric characters and space


def get_refs_from_sentence(sentence):
    return [int(citation[1:-1]) - 1 for citation in re.findall(r"\[\d+\]", sentence)]  # 0-indexed instead of 1-indexed like in text


def obj_to_filename(obj):
    return "__".join([f"{k}={v}" for k, v in obj.items()]) + ".jsonl"


def filename_to_obj(filename):
    assert filename.endswith(".jsonl")
    filename = filename.split("/")[-1][:-6]
    return {k: v for k, v in [kv.split("=") for kv in filename.split("__")]}


# remove column from dataframe that is used as index
def remove_index(df, column_name):
    keep_indexes = list(df.index.names)
    keep_indexes.remove(column_name)
    return df.reset_index().set_index(keep_indexes)


def is_object_subset(subset, superset):
    return all([superset[k] == v for k, v in subset.items()])


def create_prompt(row, prompt_id):
    system_prompt = PROMPTS[prompt_id].replace("\n", " ").strip()

    user_prompt = ""
    for i, (title, sentences) in enumerate(zip(row["context"]["title"], row["context"]["sentences"])):
        user_prompt += f"Document [{i+1}](Title: {title}): {merge_sentences(sentences)}\n"
    user_prompt += f"Question: {row['question']}"

    return user_prompt, system_prompt


def add_chatml_support(model, tokenizer):
    # tokenizer.padding_side = "left" # controversial
    if tokenizer.pad_token == None:
        print("setting pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is not None:
        print("chat template already set")
        return model, tokenizer

    PAD_TOKEN = "</s>"
    BOS_TOKEN = "<|im_start|>"
    EOS_TOKEN = "<|im_end|>"

    # todo: maybe check if these tokens are already in the tokenizer
    print(f"old bos token: {tokenizer.bos_token}, old eos token: {tokenizer.eos_token}, old pad token: {tokenizer.pad_token}")

    # tokenizer.pad_token = PAD_TOKEN
    tokenizer.add_tokens([BOS_TOKEN])
    tokenizer.add_special_tokens(dict(eos_token=EOS_TOKEN))

    # ChatML template, from https://huggingface.co/docs/transformers/main/chat_templating
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_max_memory(margin_percent=0.2):
    return None
    print("WARNING: using hardcoded max memory mapping")
    return {
        "cpu": "100GiB",
        0: "10GiB",
    }
    # todo: auto calculate
