import os
from typing import Tuple, Any

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage, ChatCompletionResponse

from aisyng.base.llms.base import LLMProvider


api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = MistralClient(api_key=api_key)

class MistralProvider(LLMProvider):
    def query_model(
            self,
            query: str,
            model: str = "mistral-large-latest",
            temperature: float = 0.1,
            **kwargs
    ) -> Tuple[str, ChatCompletionResponse]:
        chat_completion = client.chat(
            messages=[ChatMessage(role="user", content=query)],
            model=model,
            temperature=temperature,
            **kwargs
        )
        return chat_completion.choices[0].message.content, chat_completion
