"""
[역할] OpenAI Embeddings API 클라이언트
- embed_text(): 단일 텍스트 임베딩 생성
- embed_texts(): 배치 텍스트 임베딩 생성
- embed_documents(): 문서 리스트 임베딩 생성
- 텍스트를 벡터로 변환하여 RAG 및 벡터 검색에 사용
- OpenAI Embeddings API (text-embedding-3-small) 사용
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


class EmbeddingClient:
    """Wrap OpenAI embeddings to ease mocking and dependency injection."""

    def __init__(self, *, openai_client, model: str = "text-embedding-3-small") -> None:
        self.openai_client = openai_client
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        """Return a single embedding vector."""
        response = self.openai_client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Batch embedding helper."""
        if not texts:
            return []
        response = self.openai_client.embeddings.create(
            model=self.model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, documents: Iterable[dict]) -> List[dict]:
        """Attach embeddings to document payloads (mutates copies)."""
        docs = list(documents)
        if not docs:
            return []

        texts = [doc.get("content", "") for doc in docs]
        embeddings = self.embed_texts(texts)
        for doc, embedding in zip(docs, embeddings):
            doc["embedding"] = embedding
        return docs


