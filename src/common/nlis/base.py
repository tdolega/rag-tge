class NLI_BASE:
    model_name = None
    short_model_name = None

    def evaluate(self, passage: str, claim: str) -> int:
        raise NotImplementedError
