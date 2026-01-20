"""
[역할] RAG 벡터 검색 및 문서 조회
- search_similar_documents(): 벡터 유사도 검색
- retrieve_context(): 질문에 대한 관련 문서 조회
- Supabase pgvector를 사용한 벡터 유사도 검색
- 코사인 유사도 기반 문서 검색
"""

from __future__ import annotations

from functools import lru_cache
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
        verbose: bool = True,
    ) -> None:
        self.db_client = db_client
        self.embedding_client = embedding_client
        self.table_name = table_name
        self.verbose = verbose
        self._embedding_cache = {}  # 간단한 임베딩 캐시 (최근 100개)
        self._cache_max_size = 100

    def _get_cached_embedding(self, query_text: str) -> List[float]:
        """임베딩 캐싱 (속도 개선)"""
        # 캐시에 있으면 반환
        if query_text in self._embedding_cache:
            return self._embedding_cache[query_text]
        
        # 캐시에 없으면 생성
        embedding = self.embedding_client.embed_text(query_text)
        
        # 캐시 크기 제한 (LRU 방식)
        if len(self._embedding_cache) >= self._cache_max_size:
            # 가장 오래된 항목 제거 (간단하게 첫 번째 항목)
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        self._embedding_cache[query_text] = embedding
        return embedding

    def search_similar_documents(
        self,
        query_text: str,
        *,
        limit: int = 5,
        threshold: float = 0.7,
        session_name: Optional[str] = None,
        party_name: Optional[str] = None,
        agenda_title: Optional[str] = None,
        source_type: Optional[str] = None,
        use_server_search: bool = False,  # 서버 측 검색 사용 여부 (클라이언트 측 검색으로 변경)
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
        party_name : Optional[str]
            특정 정당으로 필터링
        agenda_title : Optional[str]
            특정 안건으로 필터링
        source_type : Optional[str]
            특정 소스 타입으로 필터링 (session_summary, party_position, qa_pair)
        use_server_search : bool
            서버 측 벡터 검색 사용 여부 (기본값: True)
        
        Returns
        -------
        List[Dict[str, Any]]
            검색된 문서 리스트 (content, metadata, similarity 포함)
        """
        # 질문을 벡터로 변환 (캐싱으로 속도 개선)
        query_embedding = self._get_cached_embedding(query_text)
        
        # 서버 측 벡터 검색 사용 (성능 향상)
        if use_server_search:
            try:
                return self._search_using_sql_function(
                    query_embedding=query_embedding,
                    limit=limit,
                    threshold=threshold,
                    session_name=session_name,
                    party_name=party_name,
                    agenda_title=agenda_title,
                    source_type=source_type,
                )
            except Exception as e:
                # SQL 함수가 없거나 오류 발생 시 기존 방식으로 폴백
                # (서버 측 함수가 배포되지 않은 경우 정상적인 폴백이므로 경고 숨김)
                error_str = str(e)
                if "Could not find the function" in error_str:
                    # Supabase 함수가 배포되지 않은 경우 - 정상적인 폴백, 경고 생략
                    pass
                elif self.verbose:
                    # 다른 종류의 오류는 verbose 모드에서만 표시
                    print(f"⚠️  서버 측 검색 실패, 기존 방식으로 폴백: {e}")
                use_server_search = False
        
        # 기존 방식 (클라이언트 측 계산) - 폴백
        if not use_server_search:
            return self._search_client_side(
                query_embedding=query_embedding,
                limit=limit,
                threshold=threshold,
                session_name=session_name,
                source_type=source_type,
            )
    
    def _search_using_sql_function(
        self,
        query_embedding: List[float],
        *,
        limit: int,
        threshold: float,
        session_name: Optional[str] = None,
        party_name: Optional[str] = None,
        agenda_title: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[Dict[str, any]]:
        """서버 측 SQL 함수를 사용한 벡터 검색 (성능 향상)"""
        # Supabase RPC 호출
        response = self.db_client.client.rpc(
            'search_documents_rag',
            {
                'query_embedding': query_embedding,
                'match_threshold': threshold,
                'match_count': limit,
                'filter_session_name': session_name,
                'filter_party_name': party_name,
                'filter_agenda_title': agenda_title,
                'filter_source_type': source_type,
            }
        ).execute()
        
        # 결과 변환
        results = []
        for doc in (response.data or []):
            results.append({
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "similarity": float(doc.get("similarity", 0.0)),
                "source_type": doc.get("source_type", ""),
                "source_id": doc.get("source_id", ""),
            })
        
        return results
    
    def _search_client_side(
        self,
        query_embedding: List[float],
        *,
        limit: int,
        threshold: float,
        session_name: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[Dict[str, any]]:
        """클라이언트 측 벡터 검색 (기존 방식, 폴백용)"""
        query = self.db_client.client.table(self.table_name).select("*")
        
        if source_type:
            query = query.eq("source_type", source_type)
        
        # session_name이 있으면 모든 문서를 가져오기 (배치로, 최적화)
        # session_name 필터링은 나중에 Python에서 수행 (JSON 메타데이터 필터링은 DB에서 어려움)
        documents = []
        if session_name:
            # session_name이 있으면 배치로 가져오되, 더 작은 배치로 빠르게 처리
            batch_size = 500  # 배치 크기 줄임 (속도 개선)
            max_documents = limit * 100  # 최대 문서 수 제한 (속도 개선)
            offset = 0
            matched_count = 0
            
            while len(documents) < max_documents:
                try:
                    batch = query.range(offset, offset + batch_size - 1).execute().data
                    if not batch:
                        break
                    
                    # session_name 필터링을 먼저 수행 (불필요한 유사도 계산 방지)
                    for doc in batch:
                        doc_session = doc.get("metadata", {}).get("session_name")
                        if doc_session == session_name:
                            documents.append(doc)
                            matched_count += 1
                            # 충분한 문서를 찾으면 중단 (속도 개선)
                            if matched_count >= limit * 10:  # 여유있게 가져오기
                                break
                    
                    # 충분한 문서를 찾았으면 while 루프 종료
                    if matched_count >= limit * 10:
                        break
                    
                    offset += batch_size
                    if len(batch) < batch_size:
                        break
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  배치 로딩 오류 (offset={offset}): {e}")
                    break
            
            if self.verbose:
                print(f"📚 {session_name} 회차 문서 {matched_count}개 발견 (총 {offset}개 검사)")
        else:
            # session_name이 없으면 limit * 30만 가져오기 (속도 개선: 50 -> 30)
            fetch_limit = limit * 30
            response = query.limit(fetch_limit).execute()
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
            
            # session_name 필터링을 먼저 수행 (효율성 향상)
            if session_name:
                doc_session = doc.get("metadata", {}).get("session_name")
                if doc_session != session_name:
                    continue
                # 디버깅: 매칭된 문서 수 확인
                if self.verbose and len(results) == 0 and len([r for r in results if r.get("metadata", {}).get("session_name") == session_name]) == 0:
                    # 첫 번째 매칭 문서 발견 시 로그
                    pass
            
            # 벡터를 리스트로 변환
            doc_embedding = doc["embedding"]
            if not isinstance(doc_embedding, list):
                if isinstance(doc_embedding, str):
                    import json
                    try:
                        doc_embedding = json.loads(doc_embedding)
                    except:
                        continue
                else:
                    try:
                        doc_embedding = list(doc_embedding)
                    except:
                        continue
            
            if len(doc_embedding) != len(query_embedding):
                continue
            
            similarity = cosine_similarity(query_embedding, doc_embedding)
            
            # session_name이 있으면 threshold를 더 낮춤 (해당 세션 문서 우선)
            effective_threshold = threshold * 0.5 if session_name else threshold
            if similarity < effective_threshold:
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
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
    
    def hybrid_search(
        self,
        query_text: str,
        *,
        limit: int = 5,
        threshold: float = 0.7,
        session_name: Optional[str] = None,
        party_name: Optional[str] = None,
        agenda_title: Optional[str] = None,
        source_type: Optional[str] = None,
        prefer_structured: bool = True,
    ) -> List[Dict[str, any]]:
        """하이브리드 검색: 구조화된 데이터 + 원본 발언
        
        Parameters
        ----------
        prefer_structured : bool
            구조화된 데이터 우선 사용 여부 (기본값: True)
        
        Returns
        -------
        List[Dict[str, Any]]
            검색된 문서 리스트 (구조화된 데이터 우선, 원본 발언 보완)
        """
        results = []
        
        # 1. 구조화된 데이터 검색 (우선)
        if prefer_structured:
            structured_docs = self.search_similar_documents(
                query_text,
                limit=limit * 2,  # 더 많이 가져오기
                threshold=threshold,
                session_name=session_name,
                party_name=party_name,
                agenda_title=agenda_title,
                source_type=None,  # 모든 타입
                use_server_search=True,
            )
            # 구조화된 타입만 필터링
            structured_docs = [
                doc for doc in structured_docs
                if doc.get("source_type") in [
                    "session_summary", 
                    "party_position", 
                    "qa_pair", 
                    "agenda_analysis",
                    "qa_metrics"
                ]
            ]
            results.extend(structured_docs[:limit])
        
        # 2. 원본 발언 검색 (보완)
        if len(results) < limit:
            original_docs = self.search_similar_documents(
                query_text,
                limit=limit - len(results),
                threshold=threshold * 0.9,  # 원본 발언은 임계값 약간 낮춤
                session_name=session_name,
                party_name=party_name,
                agenda_title=agenda_title,
                source_type="original_speech",
                use_server_search=True,
            )
            results.extend(original_docs)
        
        # 유사도 순으로 정렬
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
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
