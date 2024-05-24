from openai import OpenAI
from functools import cache
import random

from common.llms.base import LLM_BASE


class LLM_GPT(LLM_BASE):
    def __init__(self):
        self.client = OpenAI()

    def generate(self, user_prompt: str, system_prompt: str = ""):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.get_chat(user_prompt, system_prompt),
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            seed=random.randint(0, 2**31 - 1),  # some providers caches responses, so we need to randomize the seed
        )
        try:
            return response, response.choices[0].message.content.strip()
        except Exception as e:
            print("> response", response)
            raise e

    @cache
    def embed(self, text):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response, response.data[0].model_dump()["embedding"]
