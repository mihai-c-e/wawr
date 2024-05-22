from __future__ import annotations

import logging
import os
from enum import Enum
from typing import List, Tuple, Any, Dict, Optional, Callable

from openai import OpenAI
from openai._types import NotGiven, NOT_GIVEN

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

class InappropriateContentException(Exception):
    pass

class LLMName(str, Enum):
    OPENAI_GPT_4_TURBO = "gpt-4-turbo"
    OPENAI_GPT_35_TURBO = "gpt-3.5-turbo"

llm_provider_mapping = {
    LLMName.OPENAI_GPT_4_TURBO: 'openai',
    LLMName.OPENAI_GPT_35_TURBO: "openai"
}


class LLMProvider:
    def query_model(self, query: str, model: str, temperature: float=0.1, **kwargs) -> Tuple[str, Any]:
        raise NotImplementedError()

    def  create_embeddings(
            self, data: List[str],
            model: str, batch_size: int = 0,
            dimensions: int|NotGiven = NOT_GIVEN
    ) -> List[List[float]]:
        raise NotImplementedError()

    def get_image(self, description: str, model: str, size: str, quality: str, **kwargs):
        raise NotImplementedError()

    def moderate_text(self, text: str) -> Any:
        raise NotImplementedError()

    def query_model_threaded(
            self,
            data: List[Any],
            preprocess_fn: Optional[Callable[[Any], Any]] = None,
            model: str = LLMName.OPENAI_GPT_35_TURBO,
            temperature: float = 0.1,
            **kwargs
    ) -> List[Any]:
        raise NotImplementedError

class LLMProviderPool:
    providers: Dict[str, LLMProvider]

    def __init__(self, providers: Dict[str, LLMProvider]):
        self.providers = providers

    def get(self, provider_code: str) -> LLMProvider:
        return self.providers[provider_code]

    def get_by_model_name(self, model_name: LLMName):
        provider = llm_provider_mapping.get(model_name)
        if provider is None:
            raise ValueError(f"Provider for model '{model_name}' not found. Add it to llm_provider_mapping dict")
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} does not exist in the pool")
        return self.providers[provider]


