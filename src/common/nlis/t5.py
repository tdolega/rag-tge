from common.nlis.base import NLI_BASE
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class NLI_T5(NLI_BASE):
    def __init__(self, model_name="google/t5_xxl_true_nli_mixture"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=True)
        self.model_name = model_name.split("/")[-1]

    def evaluate(self, passage: str, claim: str) -> int:
        prompt = f"premise: {passage} hypothesis: {claim}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(input_ids, max_new_tokens=20)
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if len(result) > 1:
            result = result[0]
        if result not in ["0", "1"]:
            print(f'warning: NLI AutoAIS returned "{result}" instead of 0 or 1')
            return 0
        return int(result)
