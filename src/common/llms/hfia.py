from openai import OpenAI
import huggingface_hub

from common.llms.gpt import LLM_GPT


# * HuggingFace Inference API
class LLM_HFIA(LLM_GPT):
    def __init__(self, model_name):
        self.client = OpenAI(
            base_url=f"https://api-inference.huggingface.co/models/{model_name}/v1/",
            api_key=huggingface_hub.get_token(),
        )
