from __future__ import annotations

import json
import logging
import os
import concurrent.futures
import time
from functools import partial
from typing import List, Tuple, Any, Optional, Callable, Dict
from openai import OpenAI
from openai._types import NotGiven, NOT_GIVEN
from aisyng.base.llms.base import LLMProvider, InappropriateContentException, LLMName
from aisyng.utils import read_json

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

class OpenAIProvider(LLMProvider):
    def query_model(
            self,
            query: str,
            model: LLMName = LLMName.OPENAI_GPT_35_TURBO,
            temperature: float = 0.1,
            **kwargs
     ) -> Tuple[str, Any]:
        logging.info(f"Querying model: {model}")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model=model,
            temperature=temperature,
            **kwargs
        )
        logging.info("Response received")
        return chat_completion.choices[0].message.content, chat_completion

    def query_model_validate_json(
            self,
            query: str,
            model: str,
            temperature: float=0.1,
            retries: int = 10,
            **kwargs
    ) -> Tuple[Dict[str, Any], Any]:
        attempts = 1
        while attempts <= retries:
            try:
                result = self.query_model(query=query, model=model, temperature=temperature, **kwargs)
                result_json = read_json(result[0])
                return result_json, result[1]
            except Exception as ex:
                logging.info("Querying model for json raised exception")
                logging.exception(ex)
                if attempts == retries:
                    raise
                time.sleep(2)
                attempts += 1



    def query_model_threaded(
            self,
            data: List[Any],
            preprocess_fn: Optional[Callable[[Any], Any]] = None,
            model: LLMName = LLMName.OPENAI_GPT_35_TURBO,
            temperature: float = 0.1,
            parallelism: int=50,
            json: bool = False,
            retries: int = 5,
            **kwargs
    ) -> List[Any]:
        query_model_fn = partial(self.query_model_validate_json, retries=retries) if json else self.query_model
        if preprocess_fn is None:
            fn = partial(query_model_fn, model=model, temperature=temperature, **kwargs)
        else:
            def fn(data: Any):
                preprocessed = preprocess_fn(data)
                return query_model_fn(
                                        query=preprocessed,
                    model=model,
                    temperature=temperature,
                    **kwargs
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
            results = executor.map(fn, data)

        return list(results)


    def  create_embeddings(
            self, data: List[str],
            model: str,
            batch_size: int = 0,
            dimensions: int|NotGiven = NOT_GIVEN,
            **kwargs
    ) -> List[List[float]]:
        if batch_size <= 0:
            batch_size = len(data)
        embeddings = list()
        for i in range(0, len(data), batch_size):
            logging.info(f"Batch {i} - {i + batch_size} of {len(data)}")
            batch_embeddings = client.embeddings.create(input=data, model=model, dimensions=dimensions, **kwargs).data
            embeddings.extend(batch_embeddings)
        return [r.embedding for r in embeddings]

    def get_image(
            self,
            description: str,
            model: str = "dall-e-3",
            size: str = "1024x1024",
            quality: str = "standard",
            **kwargs
    ):
        logging.info(f"Getting image from {model} for: {description}")
        response = client.images.generate(
            model=model,
            prompt=description,
            size=size,
            quality=quality,
            n=1,
            **kwargs
        )

        image_url = response.data[0].url
        logging.info(f"Image generation for {description} complete at: {image_url}")
        return image_url

    def moderate_text(self, text: str) -> Any:
        logging.info(f"Moderating content: {text}")
        response = client.moderations.create(input=text)
        output = response.results[0]
        logging.info(f"Content: '{text}' flagged as inappropriate: {output.flagged}")
        if output.flagged:
            raise InappropriateContentException()
