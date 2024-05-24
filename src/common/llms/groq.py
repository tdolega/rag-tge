from openai import OpenAI
from dotenv import load_dotenv
import os

from common.llms.gpt import LLM_GPT

load_dotenv()


class LLM_GROQ(LLM_GPT):
    def __init__(self):
        self.client = OpenAI(
            base_url=f"https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_KEY"),
        )
