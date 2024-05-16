import os
from typing import Tuple

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage, ChatCompletionResponse

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = MistralClient(api_key=api_key)

def query_model(query: str, model="mistral-largel-latest", temperature=0.1, **kwargs) -> Tuple[str, ChatCompletionResponse]:
    chat_completion = client.chat(
        messages=[ChatMessage(role="user", content=query)],
        model=model,
        temperature=temperature,
        **kwargs
    )
    return chat_completion.choices[0].message.content, chat_completion