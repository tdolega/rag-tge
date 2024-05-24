from functools import cache
import time

from common.llms import get_llm

N_ATTEMPTS = 3
PROMPT_TEMPLATE = """
I will provide you with a premise and a hypothesis. Your task is to determine whether the hypothesis can be deduced from the premise. If the hypothesis can be deduced from the premise, respond with "True". If the hypothesis cannot be deduced from the premise, respond with "False". Do not provide any explanation, only respond with "True" or "False".

Example:
Premise: "All birds can fly. Penguins are birds. Penguins cannot fly."

Hypothesis: "Some birds cannot fly."

Response: True

Premise: "The company is releasing a new product next month. The marketing team has already started the campaign."

Hypothesis: "The product is available for purchase next month."

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
        self.model_name = self.llm.model_name
        self.short_model_name = self.llm.short_model_name

    @cache
    def evaluate(self, passage: str, claim: str) -> int:
        prompt = PROMPT_TEMPLATE.format(premise=passage, hypothesis=claim)
        for _ in range(N_ATTEMPTS):
            try:
                _, response = self.llm.generate(prompt)
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
