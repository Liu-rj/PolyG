import tiktoken
from litellm import acompletion
from litellm.types.utils import ModelResponse
from typing import List, Tuple, Dict
from transformers import AutoTokenizer


class LLM:
    def __init__(self, model: str, sampling_params: Dict = {}):
        self.model = model
        self.sampling_params = sampling_params

        if model.startswith("openai/"):
            self.token_encoder = tiktoken.get_encoding("cl100k_base")
        elif model.startswith("deepseek"):
            self.token_encoder = AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-V3.1"
            )
        elif model.startswith("hosted_vllm/"):
            tokenizer = AutoTokenizer.from_pretrained(model.lstrip("hosted_vllm/"))
            self.token_encoder = tokenizer
        else:
            raise ValueError(f"Unsupported model: {model}")

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: List[Tuple[str, str]] = [],
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Ensure history is in the correct format (user/assistant roles)
        # Assuming history_messages is [(role, content), ...]
        hist_messages = [{"role": r, "content": m} for r, m in history_messages]
        messages.extend(hist_messages)
        messages.append({"role": "user", "content": prompt})

        # Use litellm.acompletion and unpack all parameters directly.
        response = await acompletion(
            model=self.model, messages=messages, **self.sampling_params
        )

        assert isinstance(response, ModelResponse)

        return response.choices[0].message.content  # type: ignore
