import logging
import os
from typing import List

from openai import OpenAI

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

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

def create_embeddings(data: List[str], model: str, batch_size: int = 0) -> List[List[float]]:
    if batch_size <= 0:
        batch_size = len(data)
    embeddings = list()
    for i in range(0, len(data), batch_size):
        logging.info(f"Batch {i} - {i + batch_size} of {len(data)}")
        batch_embeddings = client.embeddings.create(input=data, model=model).data
        embeddings.extend(batch_embeddings)
    return [r.embedding for r in embeddings]