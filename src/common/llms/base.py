from common.utils import standarize_chat


class LLM_BASE:
    model_name = None
    short_model_name = None
    temperature = None
    max_new_tokens = 256

    def generate(self, user_prompt: str, system_prompt: str):
        raise NotImplementedError

    def embed(self, text: str):
        raise NotImplementedError

    def get_chat(self, user_prompt: str, system_prompt: str = ""):
        if system_prompt == "":
            chat = [{"role": "user", "content": user_prompt}]
        else:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return standarize_chat(self.model_name, chat)
