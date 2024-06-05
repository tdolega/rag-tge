from functools import cache
import time

from common.llms import get_llm

N_ATTEMPTS = 3
PROMPT_TEMPLATE = """
I will provide you with a premise and a hypothesis. Your task is to determine whether the hypothesis can be deduced from the premise. If the hypothesis can be deduced from the premise, respond with "True". If the hypothesis cannot be deduced from the premise, respond with "False". Do not provide any explanation, only respond with "True" or "False".

Example:
Premise: "W 1693 Edmond Halley, w oparciu o zestawienia narodzin i zgonów sporządzone przez wrocławskiego pastora Caspara Neumanna, opracował wzorzec obliczania składek emerytalnych dla powstających funduszy ubezpieczeniowych. W analizie jako miasto wzorcowe posłużył mu Wrocław."

Hypothesis: "Wrocław posłużył Halleyowi jako miasto wzorcowe w analizie składek emerytalnych."

Response: True

Premise: "Wrocław posiada trzeci po Warszawie i Krakowie pod względem wielkości dochodów (5,33 mld zł wg projektu na 2020, 5,4 mld zł wg projektu na 2021 i wydatków 5,65 mld zł w 2020 i 5,9 mld na 2021) budżet w Polsce. Dochody w przeliczeniu na jednego mieszkańca (6025 zł) ustępują jedynie Warszawie. W 2017 wartość PKB wytworzonego we Wrocławiu wyniosła 55,5 mld zł, co stanowiło 2,8% PKB Polski."

Hypothesis: "Wrocław jest trzecim najbogatszym miastem w przeliczeniu na jednego mieszkańca w Polsce."

Response: False

Your task:
Premise: "{premise}"

Hypothesis: "{hypothesis}"
"""


class NLI_LLM:
    def __init__(self, model_name):
        class DummyArgs:
            llm = model_name
            temperature = 0.1

        self.llm = get_llm(DummyArgs())
        self.llm.max_new_tokens = 16
        self.model_name = self.llm.model_name
        self.short_model_name = self.llm.short_model_name

    @cache
    def evaluate(self, passage: str, claim: str) -> int:
        prompt = PROMPT_TEMPLATE.format(premise=passage, hypothesis=claim)
        for _ in range(N_ATTEMPTS):
            try:
                print("> invoking LLM")
                _, response = self.llm.generate(prompt)
                print(f"> response: {response}")
            except Exception as e:
                print(f'error: LLM failed with exception "{e}"')
                time.sleep(1)
                continue

            decision = response.lower().strip()
            if decision.startswith("response: "):
                decision = decision[len("response: ") :]
            if decision == "true":
                return 1
            if decision == "false":
                return 0
            print(f'warning: LLM returned "{response}" parsed as "{decision}" instead of "true" or "false"')

        print(f'error: LLM failed to return "true" or "false" after {N_ATTEMPTS} attempts')
        return 0
