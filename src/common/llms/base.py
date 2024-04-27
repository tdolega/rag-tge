class LLM_BASE:
    model_name = None
    short_model_name = None
    temperature = None
    max_new_tokens = 256

    def generate(self, user_prompt: str, system_prompt: str):
        raise NotImplementedError

    def embed(self, text: str):
        raise NotImplementedError

    def get_chat(self, user_prompt: str, system_prompt: str = None):
        if system_prompt is None:
            return [{"role": "user", "content": user_prompt}]

        LLMS_WITHOUT_SYSTEM_PROMPT = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ]

        if self.model_name in LLMS_WITHOUT_SYSTEM_PROMPT:
            return [
                {"role": "user", "content": system_prompt},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": user_prompt},
            ]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
