class LLM_BASE:
    model_name = None
    temperature = None
    max_new_tokens = 1024

    def generate(self, user_prompt, system_prompt):
        raise NotImplementedError

    def embed(self, text):
        raise NotImplementedError
