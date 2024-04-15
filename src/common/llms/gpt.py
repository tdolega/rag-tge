from openai import OpenAI
from common.llms.base import LLM_BASE


class LLM_GPT(LLM_BASE):
    def __init__(self, model_name):
        self.client = OpenAI()
        self.model_name = model_name

    def generate_chat(self, user_prompt: str, system_prompt: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return response, response.choices[0].message.content.strip()

    def generate_completion(self, user_prompt: str, system_prompt: str):
        prompt = f"""
{system_prompt}

{user_prompt}
Answer:
""".strip()
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
        )
        return response, response.content.strip()

    def generate(self, user_prompt: str, system_prompt: str):
        return self.generate_chat(user_prompt, system_prompt)

    def embed(self, text):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response, response.data[0].model_dump()["embedding"]
