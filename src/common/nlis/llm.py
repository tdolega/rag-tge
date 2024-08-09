from functools import lru_cache
import time

from common.llms import get_llm
from common.consts import CACHE_SIZE

N_ATTEMPTS = 3
PROMPT_TEMPLATE = """
# Instruction:
I will provide you with a premise and a hypothesis. Your task is to determine whether the hypothesis can be deduced from the premise. If the hypothesis can be deduced from the premise, respond with "True". If the hypothesis cannot be deduced from the premise, respond with "False". Do not provide any explanation, only respond with "True" or "False".

# Example 1:
premise:
```
Stypendium może otrzymać nie więcej niż 10% studentów danego kierunku studiów. Listy rankingowe zaokrągla się w ten sposób, aby liczba otrzymujących stypendium Rektora nie przekroczyła 10% studentów danego kierunku studiów. Jeżeli liczba studentów jest mniejsza niż 10, stypendium może być przyznane 1 studentowi. Za 100% studentów uważa się wszystkich studentów danego kierunku na dzień 15 października bieżącego roku akademickiego w przypadku semestru zimowego oraz na dzień 10 marca w przypadku semestru letniego. Przy ustalaniu liczby studentów otrzymujących stypendium Rektora nie uwzględnia się studentów przyjętych na pierwszy rok studiów, o których mowa w § 12 ust. 2.

Stypendium Rektora może otrzymać student, który zajął odpowiednią pozycję na liście rankingowej oraz spełnia warunki określone w § 12 – 13.

Uprawnionych do stypendium ustala się na podstawie list rankingowych sporządzonych osobno dla każdego kierunku i stopnia studiów; listy rankingowe tworzy się w oparciu o sumę punktów stypendialnych uzyskanych za średnią ocen lub przyznanych za osiągnięcia, o których mowa w ust. 1.

Zasady sporządzania list rankingowych, punktacji osiągnięć, o których mowa w ust. 1 oraz ustalania miejsca na liście rankingowej określa załącznik nr 1.2.

Wysokość stawki stypendium ustala Rektor w porozumieniu z uczelnianym organem samorządu studenckiego, w piśmie okólnym, w terminie do 6 listopada w przypadku semestru zimowego i do 30 marca w przypadku semestru letniego.

§ 12
```
hypothesis:
```
Stypendium Rektora może otrzymać maksymalnie 10% studentów danego kierunku studiów, ale jeśli liczba studentów jest mniejsza niż 10, stypendium może zostać przyznane nawet jednemu studentowi.
```
entailment: True

# Example 2:
premise:
```
Umożliwienie wjazdu pojazdów na teren Politechniki Wrocławskiej określony w §2 ust. 1 nie jest równoznaczne z zagwarantowaniem uprawnionemu użytkownikowi wolnego miejsca postojowego do parkowania pojazdu, z wyłączeniem miejsc postojowych w garażach podziemnych oraz na parkingu wielopoziomowym w budynku C-18.

Miejsca postojowe na terenie Politechniki Wrocławskiej są niestrzeżone. Politechnika Wrocławska nie ponosi odpowiedzialności za jakiekolwiek szkody, powstałe w wyniku siły wyższej, kradzieży, zniszczenia lub uszkodzenia pojazdów znajdujących się

na terenie miejsc postojowych, jak również rzeczy w nich pozostawionych lub stanowiących ich wyposażenie.

Miejsca postojowe Politechniki Wrocławskiej z ograniczonym dostępem są możliwe do wykorzystania w godzinach od 5.30 do 22.30. W uzasadnionych wypadkach możliwe jest korzystanie z tych miejsc postojowych poza określonym wyżej przedziałem czasowym, za zgodą Kanclerza Politechniki Wrocławskiej oraz za wiedzą Działu Ochrony Mienia i Korespondencji, pod warunkiem pozostawienia numeru telefonu do właściciela lub posiadacza samochodu albo osoby upoważnionej.

O sposobie dostępu do miejsc postojowych, a także o kryteriach przydziału imiennych miejsc gwarantowanych decyduje Rektor.
```
hypothesis:
```
W uzasadnionych przypadkach można korzystać z tych miejsc postojowych poza wyznaczonym czasem, za zgodą Rektora Politechniki Wrocławskiej oraz poinformowaniem Działu Ochrony Mienia i Korespondencji, pod warunkiem podania adresu email właściciela lub użytkownika samochodu.
```
entailment: False

# Your task:
premise:
```
{premise}
```
hypothesis:
```
{hypothesis}
```
entailment:
"""


class NLI_LLM:
    def __init__(self, model_name):
        class DummyArgs:
            llm = model_name
            temperature = 0.0001

        self.llm = get_llm(DummyArgs())
        self.llm.max_new_tokens = 10
        self.model_name = self.llm.model_name
        self.short_model_name = self.llm.short_model_name

    @lru_cache(CACHE_SIZE)
    def evaluate(self, passage: str, claim: str) -> int:
        prompt = PROMPT_TEMPLATE.format(premise=passage, hypothesis=claim)
        for _ in range(N_ATTEMPTS):
            try:
                _, response = self.llm.generate(prompt)
            except Exception as e:
                print(f"error: NLI_LLM failed with exception: {e}")
                time.sleep(1)
                continue

            response = response.lower()
            if "true" in response:
                return 1
            if "false" in response:
                return 0
            print(f'warning: LLM returned "{response}" instead of "true" or "false"')

        print(f"error: NLI_LLM failed {N_ATTEMPTS} times. Returning 0.")
        return 0
