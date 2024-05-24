from common.nlis.true import NLI_TRUE


# ? used for manual verification of T5
class NLI_TRUE_INTERCEPTOR(NLI_TRUE):
    def evaluate(self, passage: str, claim: str) -> int:
        ret = super().evaluate(passage, claim)
        print("=== INTERCEPTED ===")
        print(f"passage: {passage}")
        print(f"claim: {claim}")
        print(f"result: {ret}")
        print("=== === === === ===")
        return ret
