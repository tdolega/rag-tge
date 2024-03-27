import os
import re
from datasets import load_dataset
import pandas as pd
import json

RESULTS_DIR = "../data/results/"


def get_start_idx(filename):
    if not os.path.exists(filename):
        return 0

    start_idx = 0
    with open(filename, "r") as output_handle:
        for line in output_handle:
            start_idx += 1
    print(f"resuming from index {start_idx}")
    return start_idx


def get_dataset(limit=None):
    dataset = load_dataset("hotpot_qa", "distractor", split="train")
    dataset = dataset.filter(lambda row: row["level"] == "easy", num_proc=os.cpu_count())
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
    return {k: v for kv in filename.split("__") for k, v in kv.split("=")}


def results_as_pandas(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r") as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
    data = pd.DataFrame(data)

    params = filename_to_obj(filename)
    for k, v in params.items():
        data[k] = v

    data = data.explode("evaluations")
    data = data.rename_axis("question_idx").reset_index()

    data = pd.concat([data, data["evaluations"].apply(pd.Series)], axis=1)
    evaluation_keys = data["evaluations"].apply(pd.Series).columns
    for col in evaluation_keys:
        data = pd.concat([data, data[col].apply(pd.Series).add_prefix(f"{col}/")], axis=1)
        data = data.drop(columns=col)
    data = data.drop(columns=["evaluations"])

    return data
