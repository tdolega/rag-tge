import argparse
import json
from tqdm.auto import tqdm
import time

from common.llms.llms import get_llm, add_llm_args
from common.utils import get_start_idx, get_dataset, create_prompt, obj_to_filename



def generate_answers(llm, prompt_id, n_samples, dataset_limit):
    dataset = get_dataset(dataset_limit)

    output_file = obj_to_filename(
        {
            "llm": llm.model_name,
            "prompt_id": prompt_id,
            "temperature": llm.temperature,
        }
    )
    output_file = f"../data/answers/{output_file}"
    print(f"writing into: {output_file}")
    start_idx = get_start_idx(output_file)
    output_handle = open(output_file, "a")
    for row_idx, row in tqdm(enumerate(dataset), total=len(dataset)):
        if row_idx < start_idx:
            continue
        prompts = create_prompt(row, prompt_id)
        answers = []
        for _ in range(n_samples):
            _, answer = llm.generate(*prompts)
            answers.append(answer)
        output_json = {"question_id": row["id"], "answers": answers}
        output_handle.write(json.dumps(output_json) + "\n")
        output_handle.flush()
    output_handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="generate_answers", description="Generate answers for a dataset of questions.")
    parser.add_argument("--limit", type=int, default=None, help="limit the number of questions to generate answers for; default=None")
    parser.add_argument("--prompt_id", type=int, default=1, help="prompt id from `common/prompts.py` to use; default=1")
    parser.add_argument("--n_samples", type=int, default=3, help="number of answers to generate for each question; default=3")
    add_llm_args(parser)
    args = parser.parse_args()
    print("args:", args)

    llm = get_llm(args)
    generate_answers(llm, args.prompt_id, args.n_samples, args.limit)
