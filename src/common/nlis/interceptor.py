from common.nlis.t5 import NLI_T5


# ? used for manual verification of T5
class NLI_INTERCEPTOR(NLI_T5):
    def evaluate(self, passage: str, claim: str) -> int:
        ret = super().evaluate(passage, claim)
        print("=== INTERCEPTED ===")
        print(f"passage: {passage}")
        print(f"claim: {claim}")
        print(f"result: {ret}")
        print("=== === === === ===")
        return ret
