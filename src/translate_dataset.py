import argparse
from datasets import load_dataset
from distutils.util import strtobool
from tqdm.auto import tqdm
import json
import os
import deepl
from dotenv import load_dotenv

from common.consts import DS_UPLOAD_PATH

load_dotenv()


class Cache:
    def __init__(self, backing_file):
        self.cache = {}

        assert backing_file.endswith(".jsonl"), "backing file must be a json file"
        if os.path.exists(backing_file):
            with open(backing_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    self.cache.update(data)
            print(f"> loaded {len(self.cache)} entries from cache")
        self.backing_handle = open(backing_file, "a")

    def __del__(self):
        if hasattr(self, "backing_handle"):
            self.backing_handle.close()

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, value):
        if key in self.cache:
            print(f"> warning: key '{key}' already in cache, overwriting")
        self.cache[key] = value
        self.backing_handle.write(json.dumps({key: value}, ensure_ascii=False) + "\n")
        self.backing_handle.flush()

    def __contains__(self, key):
        return key in self.cache


class Translator:
    def __init__(self, args):
        self.cache = Cache(args.cache_file)
        api_key = os.getenv("DEEPL_KEY")
        self.client = deepl.Translator(api_key)

    def __call__(self, text, context=None):
        if text in self.cache:
            return self.cache[text]

        translated = self.client.translate_text(text, target_lang="PL", context=context).text

        self.cache[text] = translated
        return translated


class Main:
    def __init__(self, args):
        self.translator = Translator(args)

    def translate_example(self, example):
        translated = {
            "prompt": self.translator(example["prompt"]),
            "answer": self.translator(example["answer"], context=example["prompt"]),
        }
        return translated

    def translate_dataset(self, args):
        dataset = load_dataset(args.dataset_name)

        print(dataset)
        if args.train_limit:
            dataset["train"] = dataset["train"].select(range(100))
        if args.test_limit:
            dataset["test"] = dataset["test"].select(range(10))
        if args.train_limit or args.test_limit:
            print(dataset)

        dataset = dataset.map(self.translate_example)
        dataset.save_to_disk(args.output_dir)
        if args.push_to_hub:
            dataset.push_to_hub(DS_UPLOAD_PATH + "_pl")
        print(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="finetune", description="Finetune a language model on a dataset of conversations.")
    boolean = lambda x: bool(strtobool(str(x)))
    parser.add_argument("--dataset_name", type=str, default=DS_UPLOAD_PATH)
    parser.add_argument("--output_dir", type=str, default="../data/training_ds_pl")
    parser.add_argument("--cache_file", type=str, default="../data/translation_cache.jsonl")
    parser.add_argument("--push_to_hub", type=boolean, default=False)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)
    args = parser.parse_args()
    print(">>> args:\n", "\n".join([f"{k}: {v}" for k, v in vars(args).items()]), "\n<<<")

    main = Main(args)
    main.translate_dataset(args)
