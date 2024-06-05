import requests
import huggingface_hub
from functools import cache

from common.sims.base import SIM_BASE


# * HuggingFace Inference API
class SIMS_HFIA(SIM_BASE):
    def __init__(self, model_name):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {huggingface_hub.get_token()}"}

    def calculate_batch(self, sentence1: str, sentences: list[str]):
        payload = {
            "inputs": {
                "source_sentence": sentence1,
                "sentences": sentences,
            },
            "options": {
                "wait_for_model": True,
            },
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        try:
            response = response.json()
            return response
        except Exception as e:
            print("> response", response)
            raise e

    @cache
    def calculate(self, sentence1: str, sentence2: str):
        response = self.calculate_batch(sentence1, [sentence2])
        return response[0]
