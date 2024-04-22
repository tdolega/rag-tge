from openai import OpenAI
from dotenv import load_dotenv
import os
import time

from common.llms.gpt import LLM_GPT

load_dotenv()


class LLM_LLMCLOUD(LLM_GPT):
    def __init__(self, model_name):
        self.model_name = model_name[len("llmcloud_") :]
        self.client = OpenAI(
            base_url="https://api.llmcloud.app/v1",
            api_key=os.getenv("LLMCLOUD_KEY"),
        )
        self.queries_ts = []
        self.queries_per_interval = 10
        self.interval_seconds = 70

    def generate_chat(self, user_prompt: str, system_prompt: str):
        if len(self.queries_ts) >= self.queries_per_interval:
            if time.time() - self.queries_ts[0] < self.interval_seconds:
                time.sleep(self.interval_seconds - (time.time() - self.queries_ts[0]))
            self.queries_ts.pop(0)

        self.queries_ts.append(time.time())

        for i in range(4):
            try:
                return super().generate_chat(user_prompt, system_prompt)
            except Exception as e:
                print(f"error {i} in LLMCloud: {e}")
                time.sleep(20)
