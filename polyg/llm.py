from litellm import completion
from litellm.types.utils import ModelResponse
from typing import List, Tuple


class LLM:
    def __init__(self, model: str):
        self.model = model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        hist_prompt: List[Tuple[str, str]] = [],
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        history_messages = [{"role": r, "content": m} for r, m in hist_prompt]

        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        response = completion(model=self.model, messages=messages, **kwargs)
        assert isinstance(response, ModelResponse)

        return response.choices[0].message.content  # type: ignore
