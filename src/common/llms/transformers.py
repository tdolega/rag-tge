from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from common.llms.base import LLM_BASE
from common.utils import get_max_memory


class LLM_TRANSFORMERS(LLM_BASE):
    def __init__(self, model_name, quantize=None, device="auto"):
        if quantize not in [None, 4, 8]:
            raise ValueError("quantize must be None, 4, or 8")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            max_memory=get_max_memory(),
            # attn_implementation="flash_attention_2",
            load_in_4bit=(quantize == 4),
            load_in_8bit=(quantize == 8),
        )

        self.terminators = [self.tokenizer.eos_token_id]
        if "Llama-3" in model_name:
            self.terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    def generate(self, user_prompt: str, system_prompt: str = None):
        inputs = self.tokenizer.apply_chat_template(self.get_chat(user_prompt, system_prompt), return_tensors="pt", add_generation_prompt=True).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                # top_p=0.9,
            )
        response = outputs[0][inputs.shape[-1] :]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        return None, response.strip()

    # todo: implement embed()
