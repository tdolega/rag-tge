import os
import json
import datasets
import pandas as pd
import argparse
from distutils.util import strtobool

from common.consts import (
    ANSWERS_DIR,
    RESULTS_DIR,
    EVAL_SIZE,
    BEST_PROMPT_ID,
    DS_SAVE_PATH,
    DS_UPLOAD_PATH,
)
from common.utils import filename_to_obj, is_object_subset, create_prompt, get_dataset


def load_all_jsonl_files(path):
    files = os.listdir(path)
    loaded = []
    for filename in files:
        with open(path + filename, "r") as handle:
            data = handle.readlines()
            data = [json.loads(line) for line in data]
            loaded.append((filename, data))
    return loaded


def satisfy_train_requirements(result):
    if result["citations"]["ais_recall"] < 1:
        return False
    if result["citations"]["ais_precision"] < 1:
        return False
    if result["correctness"]["answer_overlap"] < 0.4:
        return False
    if result["quality"]["answer_relevance"] < 0.4:
        return False
    return True


def get_training_answers():
    answers = load_all_jsonl_files(ANSWERS_DIR)
    results = load_all_jsonl_files(RESULTS_DIR)
    print(f"loaded {len(answers)} answers and {len(results)} results")
    results = [(filename, data) for filename, data in results if len(data) > EVAL_SIZE]
    print(f"filtered to {len(results)} results with more than {EVAL_SIZE} questions")

    merged = []
    for result_filename, result_data in results:
        matched_answers = []
        result_obj = filename_to_obj(result_filename)
        for answer_filename, answer_data in answers:
            answer_obj = filename_to_obj(answer_filename)
            if is_object_subset(answer_obj, result_obj):
                matched_answers.append(answer_data)
        if len(matched_answers) == 0:
            print(f"WARNING: no answer found for {result_filename}")
            continue
        if len(matched_answers) > 1:
            print(f"WARNING: found multiple answers for {result_filename}, using {matched_answers[0][0]}")
        matched_answer = matched_answers[0]
        merged.append((result_filename, result_data, matched_answer))
    print(f"merged {len(merged)} results with answers")

    training_data = []
    n_processed_questions = 0
    n_processed_examples = 0
    n_duplicate_answers = 0
    unique_answers = set()
    for result_filename, result_data, matched_answer in merged:
        for row_idx, (result_row, answer_row) in enumerate(zip(result_data, matched_answer)):
            if row_idx < EVAL_SIZE:
                continue
            n_processed_questions += 1
            assert result_row["question_id"] == answer_row["question_id"], f"Question ID mismatch at index {row_idx} for {result_filename}"
            for answer_idx, (result, answer) in enumerate(zip(result_row["evaluations"], answer_row["answers"])):
                n_processed_examples += 1
                if not satisfy_train_requirements(result):
                    continue
                if answer in unique_answers:
                    n_duplicate_answers += 1
                    continue
                unique_answers.add(answer)
                training_data.append((answer_row["question_id"], answer))

    print(f"skipped {n_duplicate_answers:_} duplicate answers")
    questions_ids = set([question_id for question_id, _ in training_data])
    examples_perc = 100 * len(training_data) / n_processed_examples
    questions_perc = 100 * len(questions_ids) / n_processed_questions
    print(f"examples kept {len(training_data):_} / {n_processed_examples:_} ({examples_perc:.2f}%)")
    print(f"questions kept {len(questions_ids):_} / {n_processed_questions:_} ({questions_perc:.2f}%)")
    return training_data


def append_questions(training_answers):
    dataset = get_dataset()
    dataset_idx_from_id = {item["id"]: idx for idx, item in enumerate(dataset)}
    training_data = []
    for question_id, answer in training_answers:
        dataset_row = dataset[dataset_idx_from_id[question_id]]
        user_prompt, system_prompt = create_prompt(dataset_row, BEST_PROMPT_ID)
        training_data.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "answer": answer,
            }
        )
    return training_data


def create_dataset(data, test_size=80):
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    dataset = dataset.shuffle(seed=50)
    dataset = dataset.train_test_split(test_size=test_size, seed=50)
    print(dataset)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    boolean = lambda x: bool(strtobool(str(x)))
    parser.add_argument("--push_to_hub", type=boolean, default=False)
    args = parser.parse_args()

    training_answers = get_training_answers()
    training_data = append_questions(training_answers)
    training_ds = create_dataset(training_data)
    training_ds.save_to_disk(DS_SAVE_PATH)
    if args.push_to_hub:
        training_ds.push_to_hub(DS_UPLOAD_PATH)
