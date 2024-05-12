from __future__ import annotations

import logging
import os
from typing import List

from openai import OpenAI
from openai._types import NotGiven, NOT_GIVEN

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

class InappropriateContentException(Exception):
    pass

def query_model(query: str, model="gpt-3.5-turbo", temperature=0.1):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model=model,
        temperature=temperature
    )
    return chat_completion.choices[0].message.content, chat_completion

def  create_embeddings(data: List[str], model: str, batch_size: int = 0, dimensions: int|NotGiven = NOT_GIVEN) -> List[List[float]]:
    if batch_size <= 0:
        batch_size = len(data)
    embeddings = list()
    for i in range(0, len(data), batch_size):
        logging.info(f"Batch {i} - {i + batch_size} of {len(data)}")
        batch_embeddings = client.embeddings.create(input=data, model=model, dimensions=dimensions).data
        embeddings.extend(batch_embeddings)
    return [r.embedding for r in embeddings]

def get_image(description: str, model: str = "dall-e-3", size: str = "1024x1024", quality: str = "standard"):
    logging.info(f"Getting image from {model} for: {description}")
    response = client.images.generate(
        model=model,
        prompt=description,
        size=size,
        quality=quality,
        n=1,
    )

    image_url = response.data[0].url
    logging.info(f"Image generation for {description} complete at: {image_url}")
    return image_url

def moderate_text(text: str):
    logging.info(f"Moderating content: {text}")
    response = client.moderations.create(input=text)
    output = response.results[0]
    logging.info(f"Content: '{text}' flagged as inappropriate: {output.flagged}")
    if output.flagged:
        raise InappropriateContentException()