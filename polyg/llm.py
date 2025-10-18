import tiktoken
import requests
import aiohttp
import time
import json
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
        else:
            tokenizer = AutoTokenizer.from_pretrained(model.lstrip("hosted_vllm/"))
            self.token_encoder = tokenizer

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

    # async def generate_w_ttft(
    #     self,
    #     prompt: str,
    #     system_prompt: str | None = None,
    #     history_messages: List[Tuple[str, str]] = [],
    # ) -> Tuple[str, int]:
    #     api_url = "http://localhost:8000/v1/completions"
    #     assert api_url.endswith(("completions", "profile")), (
    #         "OpenAI Completions API URL must end with 'completions' or 'profile'."
    #     )

    #     async with aiohttp.ClientSession(
    #         trust_env=True, timeout=aiohttp.ClientTimeout(total=6 * 60 * 60)
    #     ) as session:
    #         payload = {
    #             "model": "Qwen/Qwen3-14B",
    #             "prompt": prompt,
    #             "temperature": 0.0,
    #             "repetition_penalty": 1.0,
    #             "max_tokens": 1024,
    #             "stream": True,
    #             "stream_options": {
    #                 "include_usage": True,
    #             },
    #         }
    #         headers = {"Authorization": f"Bearer EMPTY"}

    #         # output = RequestFuncOutput()
    #         # output.prompt_len = request_func_input.prompt_len

    #         generated_text = ""
    #         st = time.perf_counter()
    #         most_recent_timestamp = st
    #         try:
    #             async with session.post(
    #                 url=api_url, json=payload, headers=headers
    #             ) as response:
    #                 if response.status == 200:
    #                     first_chunk_received = False
    #                     async for chunk_bytes in response.content:
    #                         chunk_bytes = chunk_bytes.strip()
    #                         if not chunk_bytes:
    #                             continue

    #                         chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
    #                         if chunk != "[DONE]":
    #                             data = json.loads(chunk)

    #                             # NOTE: Some completion API might have a last
    #                             # usage summary response without a token so we
    #                             # want to check a token was generated
    #                             if choices := data.get("choices"):
    #                                 # Note that text could be empty here
    #                                 # e.g. for special tokens
    #                                 text = choices[0].get("text")
    #                                 timestamp = time.perf_counter()
    #                                 # First token
    #                                 if not first_chunk_received:
    #                                     first_chunk_received = True
    #                                     ttft = time.perf_counter() - st
    #                                     # output.ttft = ttft
    #                                     print(f"[TTFT] Time to first token: {ttft:.2f}s")
    #                                     exit()

    #                                 # Decoding phase
    #                                 else:
    #                                     output.itl.append(timestamp - most_recent_timestamp)

    #                                 most_recent_timestamp = timestamp
    #                                 generated_text += text or ""
    #                             if usage := data.get("usage"):
    #                                 output.output_tokens = usage.get("completion_tokens")
    #                     if first_chunk_received:
    #                         output.success = True
    #                     else:
    #                         output.success = False
    #                         output.error = (
    #                             "Never received a valid chunk to calculate TTFT."
    #                             "This response will be marked as failed!"
    #                         )
    #                     output.generated_text = generated_text
    #                     output.latency = most_recent_timestamp - st
    #                 else:
    #                     output.error = response.reason or ""
    #                     output.success = False
    #         except Exception:
    #             output.success = False
    #             exc_info = sys.exc_info()
    #             output.error = "".join(traceback.format_exception(*exc_info))
    #     return output