from common.utils import get_chat


class LLM_BASE:
    model_name = None
    short_model_name = None
    temperature = None
    max_new_tokens = 256

    def generate(self, user_prompt: str, system_prompt: str = ""):
        raise NotImplementedError

    def embed(self, text: str):
        raise NotImplementedError

    def get_chat(self, user_prompt: str, system_prompt: str = ""):
        return get_chat(self.model_name, user_prompt, system_prompt)
