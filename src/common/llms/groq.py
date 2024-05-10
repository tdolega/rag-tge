from openai import OpenAI
import huggingface_hub

from common.llms.gpt import LLM_GPT


class LLM_GROQ(LLM_GPT):
    def __init__(self, model_name):
        raise NotImplementedError
