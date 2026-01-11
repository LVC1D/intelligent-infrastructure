from openai import OpenAI
from typing import List


class EmbeddingGenerator:
    def __init__(self):
        self.client = OpenAI()

    def embed_text(self, text: str) -> List[float]:
        if not text or text.isspace():
            raise ValueError("Text cannot be empty or whitespace")

        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )

        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # OpenAI API can embed multiple texts in one call (more efficient)
        embeds = list()

        for text in texts:
            embeds.append(self.embed_text(text))

        return embeds
