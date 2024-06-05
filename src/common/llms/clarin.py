from openai import OpenAI
from dotenv import load_dotenv
import os

from common.llms.gpt import LLM_GPT

load_dotenv()


class LLM_CLARIN(LLM_GPT):
    def __init__(self):
        self.client = OpenAI(
            base_url="https://services.clarin-pl.eu/api/v1/oapi",
            api_key=os.getenv("CLARIN_KEY"),
        )
