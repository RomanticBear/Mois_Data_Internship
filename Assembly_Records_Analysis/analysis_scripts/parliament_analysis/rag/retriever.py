"""
[역할] RAG 벡터 검색 및 문서 조회
- search_similar_documents(): 벡터 유사도 검색
- retrieve_context(): 질문에 대한 관련 문서 조회
- Supabase pgvector를 사용한 벡터 유사도 검색
- 코사인 유사도 기반 문서 검색
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..data.db_client import SupabaseDBClient
from ..data.embedding_client import EmbeddingClient


class RAGRetriever:
    """Retrieve relevant documents from vector store using semantic search."""

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

    def search_similar_documents(
        self,
        query_text: str,
        *,
        limit: int = 5,
        threshold: float = 0.7,
        session_name: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[Dict[str, any]]:
        """Search for similar documents using vector similarity.
        
        Parameters
        ----------
        query_text : str
            질문 텍스트
        limit : int
            반환할 문서 수 (기본값: 5)
        threshold : float
            최소 유사도 임계값 (0.0 ~ 1.0, 기본값: 0.7)
        session_name : Optional[str]
            특정 회차로 필터링
        source_type : Optional[str]
            특정 소스 타입으로 필터링 (session_summary, party_position, qa_pair)
        
        Returns
        -------
        List[Dict[str, Any]]
            검색된 문서 리스트 (content, metadata, similarity 포함)
        """
        # 질문을 벡터로 변환
        query_embedding = self.embedding_client.embed_text(query_text)
        
        # Supabase에서 문서 조회
        # 참고: Supabase PostgREST는 벡터 검색을 직접 지원하지 않으므로
        # 모든 문서를 가져와서 Python에서 유사도 계산
        # (더 나은 방법: Supabase에 SQL 함수를 추가하여 서버 측에서 계산)
        
        query = self.db_client.client.table(self.table_name).select("*")
        
        # 필터링
        # 참고: metadata JSONB 필터링은 PostgREST에서 다르게 처리될 수 있음
        if source_type:
            query = query.eq("source_type", source_type)
        
        # 임시로 많은 문서를 가져와서 필터링 (실제로는 SQL 함수 사용 권장)
        # session_name 필터링은 Python에서 수행
        response = query.limit(limit * 50).execute()  # 더 많이 가져와서 필터링
        
        documents = getattr(response, "data", []) or []
        
        # Python에서 코사인 유사도 계산
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            """코사인 유사도 계산"""
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        
        results = []
        for doc in documents:
            if not doc.get("embedding"):
                continue
            
            # 벡터를 리스트로 변환 (Supabase에서 가져올 때 형태가 다를 수 있음)
            doc_embedding = doc["embedding"]
            if not isinstance(doc_embedding, list):
                # 문자열이나 다른 형태일 경우 처리
                if isinstance(doc_embedding, str):
                    import json
                    try:
                        doc_embedding = json.loads(doc_embedding)
                    except:
                        continue
                else:
                    # numpy array나 다른 형태
                    try:
                        doc_embedding = list(doc_embedding)
                    except:
                        continue
            
            # 벡터 차원 확인
            if len(doc_embedding) != len(query_embedding):
                continue
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(query_embedding, doc_embedding)
            
            # 필터링
            if similarity < threshold:
                continue
            
            if session_name and doc.get("metadata", {}).get("session_name") != session_name:
                continue
            
            if source_type and doc.get("source_type") != source_type:
                continue
            
            results.append({
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "similarity": float(similarity),
                "source_type": doc.get("source_type"),
                "source_id": doc.get("source_id"),
            })
        
        # 유사도 순으로 정렬하고 limit만큼 반환
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def retrieve_context(
        self,
        query_text: str,
        *,
        top_k: int = 3,
        session_name: Optional[str] = None,
    ) -> str:
        """Retrieve context documents for RAG.
        
        Parameters
        ----------
        query_text : str
            질문 텍스트
        top_k : int
            가져올 문서 수 (기본값: 3)
        session_name : Optional[str]
            특정 회차로 필터링
        
        Returns
        -------
        str
            검색된 문서들을 조합한 컨텍스트 텍스트
        """
        documents = self.search_similar_documents(
            query_text,
            limit=top_k,
            session_name=session_name,
        )
        
        context_parts = []
        for idx, doc in enumerate(documents, start=1):
            metadata = doc.get("metadata", {})
            source_type = doc.get("source_type", "")
            
            # 메타데이터 정보 추가
            header = f"[문서 {idx}]"
            if source_type == "session_summary":
                header += f" 세션 요약: {metadata.get('session_name', '')}"
            elif source_type == "party_position":
                header += f" 정당 입장: {metadata.get('party_name', '')} - {metadata.get('agenda_title', '')}"
            elif source_type == "qa_pair":
                header += f" 질의응답: {metadata.get('questioner', '')} → {metadata.get('respondent', '')}"
            
            context_parts.append(f"{header}\n{doc['content']}\n")
        
        return "\n".join(context_parts)

