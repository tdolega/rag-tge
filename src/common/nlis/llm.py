from common.nlis.base import NLI_BASE


class NLI_LLM(NLI_BASE):
    def __init__(self, model_name="llama.cpp"):
        self.model_name = model_name
        raise NotImplementedError("LLM evaluator not implemented yet")

    def generate(self, user_prompt, system_prompt):
        raise NotImplementedError("LLM evaluator not implemented yet")
