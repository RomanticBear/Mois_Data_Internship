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
            print("  ⚠️  저장할 RAG 문서가 없습니다.")
            return

        print(f"  📝 {len(vector_items)}개 RAG 문서 처리 중...")

        items_without_embedding = [item for item in vector_items if item.embedding is None]
        if items_without_embedding:
            print(f"  🔄 {len(items_without_embedding)}개 문서 임베딩 생성 중...")
            try:
                embeddings = self.embedding_client.embed_texts(
                    [item.content for item in items_without_embedding]
                )
                for item, embedding in zip(items_without_embedding, embeddings):
                    item.embedding = embedding
                print(f"  ✅ 임베딩 생성 완료")
            except Exception as e:
                print(f"  ❌ 임베딩 생성 실패: {e}")
                import traceback
                traceback.print_exc()
                raise

        payload = []
        for item in vector_items:
            metadata = dict(item.metadata)
            source_id = metadata.get("source_id")
            chunk_index = metadata.get("chunk_index")
            
            # 임베딩에서 NaN 값 제거
            embedding = item.embedding
            if embedding is not None:
                import math
                embedding = [
                    float(val) if not (math.isnan(val) or math.isinf(val)) else 0.0
                    for val in embedding
                ]
            
            payload.append(
                {
                    "source_type": metadata.get("source_type"),
                    "source_id": source_id,
                    "chunk_index": chunk_index,
                    "content": item.content,
                    "embedding": embedding,
                    "metadata": metadata,
                }
            )
        
        print(f"  💾 {len(payload)}개 문서 DB 저장 중...")
        try:
            self.db_client.upsert_rag_documents(payload)
            print(f"  ✅ RAG 문서 저장 완료: {len(payload)}개")
        except Exception as e:
            print(f"  ❌ RAG 문서 저장 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

    def delete_documents_by_source(self, *, source_id: str) -> None:
        """Remove outdated documents for a given source."""
        self.db_client.delete_rag_documents_by_source(source_id=source_id)


