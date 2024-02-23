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

def create_embeddings(data: List[str], model: str) -> List[List[float]]:
    result = client.embeddings.create(input=data, model=model).data
    return [r.embedding for r in result]