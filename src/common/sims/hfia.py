import requests
import huggingface_hub
from functools import cache

from common.sims.base import SIM_BASE


# * HuggingFace Inference API
class SIMS_HFIA(SIM_BASE):
    def __init__(self, model_name):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {huggingface_hub.get_token()}"}

    @cache
    def calculate(self, sentence1: str, sentence2: str):
        payload = {
            "inputs": {
                "source_sentence": sentence1,
                "sentences": [sentence2],
            },
            "options": {
                "wait_for_model": True,
            },
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        try:
            response = response.json()
            return response, response[0]
        except Exception as e:
            print("> response", response)
            print("> response.json()", response.json())
            raise e
