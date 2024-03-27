class LLM_BASE:
    model_name = None
    temperature = None

    def generate(self, user_prompt, system_prompt):
        raise NotImplementedError

    def embed(self, text):
        raise NotImplementedError
