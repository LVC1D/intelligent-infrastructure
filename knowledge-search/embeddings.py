from openai import OpenAI
from typing import List
import re


class EmbeddingGenerator:
    def __init__(self):
        self.client = OpenAI()
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_max_size = 100

    def embed_text(self, text: str) -> List[float]:
        if not text or text.isspace():
            raise ValueError("Text cannot be empty or whitespace")

        norm_text = self._normalize_text(text)

        if norm_text in self.cache: 
            self.cache_hits += 1
            return self.cache[norm_text]

        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )

        embed_vector = response.data[0].embedding 

        self.cache[norm_text] = embed_vector
        self.cache_misses += 1

        if len(self.cache) > self.cache_max_size:
            # Remove oldest entry 
            self.cache.pop(next(iter(self.cache)))

        return embed_vector

    def _normalize_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.lower().strip())

    def get_cache_stats(self) -> dict:
        stats = dict()
        stats["hits"] = self.cache_hits
        stats["misses"] = self.cache_misses
        stats["size"] = len(self.cache)
        return stats


    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # OpenAI API can embed multiple texts in one call (more efficient)
        embeds = list()

        for text in texts:
            embeds.append(self.embed_text(text))

        return embeds
