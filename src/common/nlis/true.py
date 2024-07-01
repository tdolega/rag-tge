from common.nlis.base import NLI_BASE
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from functools import lru_cache

from common.utils import get_max_memory
from common.consts import CACHE_SIZE


class NLI_TRUE(NLI_BASE):
    def __init__(self, model_name, device="auto"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            max_memory=get_max_memory(),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=True)

    @lru_cache(CACHE_SIZE)
    def evaluate(self, passage: str, claim: str) -> int:
        # cut the input if it's too long, but try to keep the claim
        MAX_LENGTH = 512
        SAFE_BUFFER = 8
        passage_tokens = self.tokenizer(passage, return_tensors="pt").input_ids
        claim_tokens = self.tokenizer(claim, return_tensors="pt").input_ids
        passage_n_tokens = passage_tokens.size(1)
        claim_n_tokens = claim_tokens.size(1)
        if passage_n_tokens + claim_n_tokens + SAFE_BUFFER > MAX_LENGTH:
            if claim_n_tokens > MAX_LENGTH // 2:
                claim_tokens = claim_tokens[:, : MAX_LENGTH // 2]
                claim = self.tokenizer.decode(claim_tokens[0], skip_special_tokens=True)
                claim_n_tokens = claim_tokens.size(1)
            if passage_n_tokens + claim_n_tokens + SAFE_BUFFER > MAX_LENGTH:
                passage_tokens = passage_tokens[:, : MAX_LENGTH - claim_n_tokens - SAFE_BUFFER]
                passage = self.tokenizer.decode(passage_tokens[0], skip_special_tokens=True)
                passage_n_tokens = passage_tokens.size(1)

        prompt = f"premise: {passage} hypothesis: {claim}"
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).input_ids
        input_ids = input_ids.to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(input_ids, max_new_tokens=10)
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if len(result) > 1:
            result = result[0]
        if result not in ["0", "1"]:
            print(f'warning: NLI AutoAIS returned "{result}" instead of 0 or 1')
            return 0
        return int(result)
