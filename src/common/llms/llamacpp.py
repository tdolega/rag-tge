from openai import OpenAI

from common.llms.gpt import LLM_GPT


class LLM_LLAMACPP(LLM_GPT):
    def __init__(self):
        self.client = OpenAI(
            # base_url="http://10.9.1.208:51200/v1",
            # base_url="http://10.9.1.208:51201/v1",
            base_url="http://localhost:51201/v1",
            api_key="sk-pnw2024",  # not a secret
        )
        models = self.client.models.list()
        self.model_name = models.data[0].id.split("/")[-1]
        if self.model_name.endswith(".gguf"):
            self.model_name = self.model_name[: -len(".gguf")]
