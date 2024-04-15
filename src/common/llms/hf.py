from transformers import AutoTokenizer, pipeline
import torch

from common.llms.base import LLM_BASE


class LLM_HF(LLM_BASE):
    def __init__(self, model_name, quantize=8, device="auto"):
        if quantize not in [None, 4, 8]:
            raise ValueError("quantize must be None, 4, or 8")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = pipeline(
            task="text-generation",
            model=model_name,
            tokenizer=tokenizer,
            device_map=device,
            torch_dtype=torch.bfloat16,
            num_return_sequences=1,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=self.temperature,
            repetition_penalty=1.1,
            trust_remote_code=True,
            do_sample=True,
            # top_k=10,
            max_new_tokens=self.max_new_tokens,
            model_kwargs={
                "load_in_4bit": quantize == 4,
                "load_in_8bit": quantize == 8,
            },
        )
        model_name = model_name.split("/")[-1]
        self.model_name = f"{model_name}" + (f"-Q{quantize}" if quantize else "")

    def generate(self, user_prompt, system_prompt):
        if system_prompt:
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        else:
            prompt = f"[INST] {user_prompt} [/INST]"
        with torch.inference_mode():
            output = self.model(prompt)
        return output, output[0]["generated_text"]

    # todo: implement embed()
