from openai import OpenAI

from common.llms.gpt import LLM_GPT


class LLM_LLAMACPP(LLM_GPT):
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:51201/v1",
            api_key="sk-pnw2024",  # not a secret
        )
        models = self.client.models.list()
        self.model_name = models.data[0].id
        self.short_model_name = self.model_name.split("/")[-1]
        if self.short_model_name.endswith(".gguf"):
            self.short_model_name = self.short_model_name[: -len(".gguf")]
