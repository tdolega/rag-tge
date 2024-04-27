class SIM_BASE:
    model_name = None
    short_model_name = None

    def calculate(self, sentence1: str, sentence2: str):
        raise NotImplementedError
