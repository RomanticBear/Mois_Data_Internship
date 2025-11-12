"""
[역할] 벡터 스토어 래퍼
- upsert_documents(): 벡터 문서 저장 (임베딩 생성 포함)
- delete_documents_by_source(): 특정 소스의 문서 삭제
- EmbeddingClient를 사용하여 텍스트를 벡터로 변환
- Supabase documents_rag 테이블에 벡터 및 메타데이터 저장
- RAG 검색을 위한 벡터 저장소 관리
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from ..data.db_client import SupabaseDBClient
from ..data.embedding_client import EmbeddingClient


@dataclass
class VectorItem:
    record_id: str
    content: str
    metadata: Dict[str, object]
    embedding: Optional[List[float]] = None


class VectorStore:
    """Attach embeddings and persist them to the documents_rag table."""

    def __init__(
        self,
        *,
        db_client: SupabaseDBClient,
        embedding_client: EmbeddingClient,
        table_name: str = "documents_rag",
    ) -> None:
        self.db_client = db_client
        self.embedding_client = embedding_client
        self.table_name = table_name

    def upsert_documents(self, items: Iterable[VectorItem]) -> None:
        """Ensure each vector item has an embedding and is stored."""
        vector_items = list(items)
        if not vector_items:
            return

        items_without_embedding = [item for item in vector_items if item.embedding is None]
        if items_without_embedding:
            embeddings = self.embedding_client.embed_texts(
                [item.content for item in items_without_embedding]
            )
            for item, embedding in zip(items_without_embedding, embeddings):
                item.embedding = embedding

        payload = []
        for item in vector_items:
            metadata = dict(item.metadata)
            source_id = metadata.get("source_id")
            chunk_index = metadata.get("chunk_index")
            payload.append(
                {
                    "source_type": metadata.get("source_type"),
                    "source_id": source_id,
                    "chunk_index": chunk_index,
                    "content": item.content,
                    "embedding": item.embedding,
                    "metadata": metadata,
                }
            )
        self.db_client.upsert_rag_documents(payload)

    def delete_documents_by_source(self, *, source_id: str) -> None:
        """Remove outdated documents for a given source."""
        self.db_client.delete_rag_documents_by_source(source_id=source_id)


