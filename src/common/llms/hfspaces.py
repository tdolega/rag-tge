from gradio_client import Client

from common.llms.base import LLM_BASE


# ? support for random HuggingFace spaces models, not standard Pro spaces
class LLM_HFSPACES(LLM_BASE):
    def __init__(self, model_name):
        model_name = model_name[len("hfspaces_") :]
        self.client = Client(model_name)
        self.model_name = model_name.split("/")[1]

    ## Qwen
    def generate(self, user_prompt: str, system_prompt: str):
        response = self.client.predict(
            system=system_prompt,
            query=user_prompt,
            history=[],
            api_name="/model_chat",
        )
        return response, response[1][0][1].strip()

    ## llama
    # def generate(self, user_prompt: str, system_prompt: str):
    #     response = self.client.predict(
    #         request=system_prompt,
    #         message=user_prompt,
    #         api_name="/chat",
    #         param_4=self.temperature,
    #     )
    #     return response, response.strip()
