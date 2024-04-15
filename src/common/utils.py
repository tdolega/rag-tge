import os
import re
from datasets import load_dataset

from common.prompts import PROMPTS

RESULTS_DIR = "../data/results/"
ANSWERS_DIR = "../data/answers/"
EVAL_SIZE = 100


def get_start_idx(filename):
    if not os.path.exists(filename):
        return 0

    start_idx = 0
    with open(filename, "r") as output_handle:
        for line in output_handle:
            start_idx += 1
    print(f"resuming from index {start_idx}")
    return start_idx


def get_dataset(limit=None, split="train", level="easy"):
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    if level is not None:
        dataset = dataset.filter(lambda row: row["level"] == level, num_proc=os.cpu_count())
    dataset = dataset.shuffle(seed=102)
    if limit is not None:
        dataset = dataset.select(range(limit))
    return dataset


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

