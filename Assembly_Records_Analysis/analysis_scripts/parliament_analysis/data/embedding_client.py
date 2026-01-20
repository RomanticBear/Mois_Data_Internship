"""
[역할] OpenAI Embeddings API 클라이언트
- embed_text(): 단일 텍스트 임베딩 생성
- embed_texts(): 배치 텍스트 임베딩 생성
- embed_documents(): 문서 리스트 임베딩 생성
- 텍스트를 벡터로 변환하여 RAG 및 벡터 검색에 사용
- OpenAI Embeddings API (text-embedding-3-small) 사용
- Rate limit 및 429 오류 처리 포함
"""

from __future__ import annotations

import os
import time
from typing import Iterable, List, Sequence

from openai import RateLimitError, AuthenticationError


class EmbeddingClient:
    """Wrap OpenAI embeddings to ease mocking and dependency injection."""

    def __init__(
        self,
        *,
        openai_client,
        model: str = "text-embedding-3-small",
        max_retries: int = 3,
        base_delay: float = 1.0,
        request_delay: float = 0.1,  # 요청 간 최소 딜레이 (초)
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.request_delay = request_delay
        self.last_request_time = 0.0

    def _wait_for_rate_limit(self):
        """Rate limit 방지를 위한 요청 간 딜레이"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_delay:
            time.sleep(self.request_delay - time_since_last_request)
        self.last_request_time = time.time()

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Exponential backoff를 사용한 재시도 로직"""
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                return func(*args, **kwargs)
            except AuthenticationError as e:
                # 401 오류는 재시도해도 의미 없음 - 즉시 실패
                print("❌ API 인증 실패: API 키가 올바르지 않거나 만료되었습니다.")
                print("   다음을 확인하세요:")
                print("   1. .env 파일의 OPENAI_API_KEY가 올바른지 확인")
                print("   2. API 키가 'sk-'로 시작하는지 확인")
                print("   3. API 키에 공백이나 따옴표가 포함되지 않았는지 확인")
                print("   4. OpenAI 대시보드에서 키가 활성화되어 있는지 확인")
                print(f"   키 시작 부분: {os.getenv('OPENAI_API_KEY', '')[:7] if os.getenv('OPENAI_API_KEY') else 'N/A'}...")
                raise
            except RateLimitError as e:
                # 상세한 오류 정보 출력
                error_info = {}
                if hasattr(e, 'response') and e.response is not None:
                    error_info['status_code'] = e.response.status_code
                    error_info['headers'] = dict(e.response.headers) if e.response.headers else {}
                    
                    # 응답 본문 확인
                    if hasattr(e.response, 'json'):
                        try:
                            error_info['body'] = e.response.json()
                        except:
                            error_info['body'] = str(e.response.text) if hasattr(e.response, 'text') else None
                
                # 오류 메시지에서 상세 정보 추출
                error_body = error_info.get('body', {})
                if isinstance(error_body, dict):
                    error_type = error_body.get('error', {}).get('type', 'unknown')
                    error_code = error_body.get('error', {}).get('code', 'unknown')
                    error_message = error_body.get('error', {}).get('message', str(e))
                    
                    print(f"\n🔍 429 오류 상세 정보:")
                    print(f"   오류 타입: {error_type}")
                    print(f"   오류 코드: {error_code}")
                    print(f"   메시지: {error_message}")
                    
                    # retry-after 헤더 확인
                    retry_after = error_info.get('headers', {}).get('retry-after')
                    if retry_after:
                        print(f"   Retry-After: {retry_after}초")
                    
                    # insufficient_quota인 경우 특별 처리
                    if error_code == 'insufficient_quota' or error_type == 'insufficient_quota':
                        print(f"\n❌ 할당량 부족 (Quota Exceeded)")
                        print(f"   원인 분석:")
                        print(f"   1. 계정 잔액 부족 가능성")
                        print(f"   2. 결제 수단 미등록 가능성")
                        print(f"   3. 월간/일일 사용 한도 초과 가능성")
                        print(f"   4. 새 계정의 초기 제한 가능성")
                        print(f"\n   확인 방법:")
                        print(f"   - https://platform.openai.com/account/billing (잔액 확인)")
                        print(f"   - https://platform.openai.com/account/limits (제한 확인)")
                        print(f"   - https://platform.openai.com/account/usage (사용량 확인)")
                        # 할당량 부족은 재시도해도 의미 없음
                        raise
                
                if attempt == self.max_retries - 1:
                    raise
                
                # Exponential backoff: 1초, 2초, 4초...
                delay = self.base_delay * (2 ** attempt)
                
                # retry-after 헤더 확인
                retry_after = error_info.get('headers', {}).get('retry-after')
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        pass
                
                print(f"⚠️  Rate limit 초과 (시도 {attempt + 1}/{self.max_retries}). {delay:.1f}초 후 재시도...")
                time.sleep(delay)
            except Exception as e:
                # Rate limit이 아닌 다른 오류는 즉시 재발생
                raise

    def embed_text(self, text: str) -> List[float]:
        """Return a single embedding vector with retry logic."""
        def _call_api():
            return self.openai_client.embeddings.create(
            model=self.model,
            input=text,
        )
        
        response = self._retry_with_backoff(_call_api)
        return response.data[0].embedding

    def embed_texts(self, texts: Sequence[str], batch_size: int = 100) -> List[List[float]]:
        """Batch embedding helper with retry logic.
        
        Args:
            texts: 텍스트 리스트
            batch_size: 한 번에 처리할 텍스트 수 (토큰 제한 방지)
        """
        if not texts:
            return []
        
        # 배치로 나누어 처리 (토큰 제한 방지)
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            def _call_api():
                return self.openai_client.embeddings.create(
                    model=self.model,
                    input=list(batch),
                )
            
            response = self._retry_with_backoff(_call_api)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # 진행 상황 출력 (큰 배치의 경우)
            if len(texts) > 200:
                print(f"    임베딩 진행: {min(i + batch_size, len(texts))}/{len(texts)}")
        
        return all_embeddings

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


