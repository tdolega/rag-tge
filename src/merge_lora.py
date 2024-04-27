from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import argparse
from distutils.util import strtobool
from dotenv import load_dotenv

from common.utils import add_chatml_support

load_dotenv()


def merge(adapter_path, push_to_hub):
    if adapter_path.endswith("/"):
        adapter_path = adapter_path[:-1]
    assert adapter_path.endswith("_LoRA")
    output_path = adapter_path[: -len("_LoRA")]

    peft_config = PeftConfig.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model, tokenizer = add_chatml_support(model, tokenizer)

    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path, push_to_hub=push_to_hub)

    tokenizer.save_pretrained(output_path, push_to_hub=push_to_hub)

    print(f"saved model and tokenizer to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    boolean = lambda x: bool(strtobool(str(x)))
    parser.add_argument("adapter_path", type=str)
    parser.add_argument("--push_to_hub", type=boolean, default=False)
    args = parser.parse_args()
    merge(**vars(args))
