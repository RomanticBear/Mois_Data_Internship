"""
[역할] RAG 기반 질문-답변 시스템
- ask_question(): 질문을 받아서 벡터 DB에서 관련 문서를 검색하고 답변 생성
- RAGRetriever를 사용하여 관련 문서 검색
- OpenAI LLM을 사용하여 컨텍스트 기반 답변 생성
- Rate limit 및 429 오류 처리 포함
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from openai import OpenAI, RateLimitError

from .retriever import RAGRetriever


class RAGQASystem:
    """RAG-based Question-Answering system for parliamentary records."""

    def __init__(
        self,
        *,
        retriever: RAGRetriever,
        llm_client: OpenAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_retries: int = 3,
        base_delay: float = 1.0,
        request_delay: float = 0.1,  # 요청 간 최소 딜레이 (초) - 속도 개선
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
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
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise
                
                # Exponential backoff: 1초, 2초, 4초...
                delay = self.base_delay * (2 ** attempt)
                
                # 429 오류 메시지에서 retry-after 정보 확인
                if hasattr(e, 'response') and e.response is not None:
                    retry_after = e.response.headers.get('retry-after')
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

    def _route_question(self, question: str) -> Dict[str, any]:
        """LLM 라우터: 질문이 검색이 필요한지 판단
        
        Returns
        -------
        Dict[str, any]
            {
                "needs_search": bool,
                "route_type": str,  # 'search', 'general', 'clarify'
                "reason": str
            }
        """
        question_lower = question.lower().strip()
        
        # 1차 필터링: 명확히 검색 불필요한 일반 질문/메타 질문
        no_search_keywords = [
            '안녕', 'hello', 'hi', '헬로',
            '너 뭐', '너는', '너가', '너의',
            '뭐 하는', '무엇을 하는', '뭐야', '무엇이야',
            '검색이', '왜 이렇게', '어떻게 작동', '어떻게 동작',
            '질문 예시', '예시', '예를', '예제',
            'rag가', 'rag는', 'rag이', 'rag이 뭐',
            '챗봇이', '챗봇은', '챗봇의',
            '도움', '도와', 'help', 'help me',
            '고마워', '감사', 'thanks', 'thank you',
        ]
        
        # 회의록 관련 키워드 (검색 필요)
        search_keywords = [
            '회의', '회의록', '본회의', '위원회',
            '안건', '쟁점', '논의', '발언',
            '의원', '위원', '정당', '입장',
            '제', '회차', '회', '차',
            '누가', '언제', '어디서', '왜', '어떻게',
            '요약', '주요', '핵심', '내용',
        ]
        
        # 명확히 검색 불필요한 경우
        if any(keyword in question_lower for keyword in no_search_keywords):
            if not any(keyword in question_lower for keyword in search_keywords):
                return {
                    "needs_search": False,
                    "route_type": "general",
                    "reason": "일반 질문/메타 질문으로 판단"
                }
        
        # 회의록 관련 키워드가 있으면 검색 필요
        if any(keyword in question_lower for keyword in search_keywords):
            return {
                "needs_search": True,
                "route_type": "search",
                "reason": "회의록 관련 질문으로 판단"
            }
        
        # 추가 키워드 기반 필터링 강화 (LLM 호출 최소화)
        # 질문이 너무 짧거나 모호한 경우
        if len(question_lower.strip()) < 3:
            return {
                "needs_search": False,
                "route_type": "clarify",
                "reason": "질문이 너무 짧음"
            }
        
        # 일반적인 질문 패턴 (검색 불필요)
        general_patterns = [
            '뭐야', '뭐', '무엇', '어떻게', '왜', '언제', '어디',
            '설명', '알려', '가르쳐', '도와', 'help'
        ]
        if any(pattern in question_lower for pattern in general_patterns):
            if not any(keyword in question_lower for keyword in search_keywords):
                return {
                    "needs_search": False,
                    "route_type": "general",
                    "reason": "일반 질문 패턴으로 판단"
                }
        
        # 애매한 경우: LLM으로 판단 (간소화된 프롬프트로 속도 개선)
        routing_prompt = f"""질문: {question}

다음 중 하나로 분류하세요:
- search: 회의록 내용 질문 (회차, 안건, 발언 등)
- general: 챗봇 사용법/인사말
- clarify: 모호한 질문

JSON: {{"needs_search": true/false, "route_type": "search|general|clarify", "reason": "간단한 이유"}}"""

        try:
            def _call_llm():
                return self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "질문 라우팅 전문가입니다. JSON 형식으로만 응답하세요.",
                        },
                        {"role": "user", "content": routing_prompt},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    max_tokens=150,  # 속도 개선: 토큰 수 제한
                )
            
            response = self._retry_with_backoff(_call_llm)
            result_text = response.choices[0].message.content
            
            import json
            routing_result = json.loads(result_text)
            
            print(f"🔀 라우팅 결과: {routing_result.get('route_type')} - {routing_result.get('reason')}")
            return routing_result
            
        except Exception as e:
            print(f"⚠️  라우팅 실패, 기본값 사용: {str(e)}")
            # 실패 시 회의록 관련 키워드가 있으면 검색 필요로 판단
            if any(keyword in question_lower for keyword in search_keywords):
                return {
                    "needs_search": True,
                    "route_type": "search",
                    "reason": "라우팅 실패, 키워드 기반 판단"
                }
            else:
                return {
                    "needs_search": False,
                    "route_type": "general",
                    "reason": "라우팅 실패, 일반 질문으로 판단"
                }
    
    def _handle_general_question(self, question: str, reason: str) -> Dict[str, any]:
        """일반 질문/메타 질문 처리 (검색 없이 LLM이 직접 답변)"""
        general_prompt = f"""국회 회의록 챗봇 안내자입니다. 다음 질문에 높임말로 친절하게 답변해주세요.

질문: {question}

답변 지침:
- 높임말 사용 필수 (반말 금지)
- 챗봇 목적: 국회 회의록 데이터 기반 RAG 시스템
- 기능: 회차별 논의 내용, 안건 쟁점, 발언자 정보 검색
- 예시: "제415회 주요 논의 내용이 뭐였어?"
- 간결하고 친절하게 답변
"""

        def _call_llm():
            return self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "국회 회의록 챗봇 안내자입니다. "
                            "반드시 높임말을 사용하여 친절하고 간결하게 답변하세요."
                        ),
                    },
                    {"role": "user", "content": general_prompt},
                ],
                temperature=self.temperature,
                max_tokens=300,  # 속도 개선: 토큰 수 제한
            )
        
        try:
            response = self._retry_with_backoff(_call_llm)
            answer_text = response.choices[0].message.content
            
            return {
                "answer": answer_text,
                "sources": [],
                "question": question,
            }
        except Exception as e:
            print(f"⚠️  일반 질문 처리 실패: {str(e)}")
            return {
                "answer": (
                    "안녕하세요! 저는 국회 회의록 챗봇입니다. "
                    "특정 회차의 논의 내용, 안건 쟁점, 발언자 정보 등을 검색할 수 있습니다. "
                    "구체적인 질문을 해주시면 도와드리겠습니다.\n\n"
                    "예시: '제415회 주요 논의 내용이 뭐였어요?'"
                ),
                "sources": [],
                "question": question,
            }
    
    def ask_question(
        self,
        question: str,
        *,
        session_name: Optional[str] = None,
        top_k: int = 3,
        include_sources: bool = True,
    ) -> Dict[str, any]:
        """질문에 대한 답변을 생성합니다.
        
        Parameters
        ----------
        question : str
            사용자 질문
        session_name : Optional[str]
            특정 회차로 필터링 (None이면 전체 검색)
        top_k : int
            검색할 문서 수 (기본값: 3)
        include_sources : bool
            출처 정보 포함 여부
        
        Returns
        -------
        Dict[str, Any]
            {
                "answer": "답변 텍스트",
                "sources": [검색된 문서 리스트],
                "question": "원본 질문",
            }
        """
        # 1. 질문 라우팅: 검색이 필요한지 판단
        routing = self._route_question(question)
        
        # 검색 불필요한 일반 질문 처리
        if routing["route_type"] == "general":
            print(f"💬 일반 질문으로 판단: {routing['reason']}")
            return self._handle_general_question(question, routing["reason"])
        
        # 명확화 필요한 질문 처리
        if routing["route_type"] == "clarify":
            print(f"❓ 명확화 필요: {routing['reason']}")
            return {
                "answer": (
                    "질문이 너무 모호합니다. 다음 정보를 포함해서 다시 질문해주세요:\n"
                    "- 회차 번호 (예: 제415회)\n"
                    "- 위원회 또는 안건 정보\n"
                    "- 구체적인 질문 내용\n\n"
                    "예시: '제415회 본회의에서 주요 논의 내용이 뭐였어요?'"
                ),
                "sources": [],
                "question": question,
            }
        
        # 검색 필요한 질문 처리
        print(f"🔍 검색 필요: {routing['reason']}")
        print(f"🔍 질문: {question}")
        
        # session_name이 없으면 질문에서 회차 추출 시도
        if not session_name:
            import re
            # "424회", "제424회", "424 회" 등의 패턴 추출
            session_match = re.search(r'제?\s*(\d+)\s*회', question)
            if session_match:
                session_num = session_match.group(1)
                session_name = f"제{session_num}회"
                print(f"📌 질문에서 회차 자동 추출: {session_name}")
        
        if session_name:
            print(f"📌 필터링: {session_name} 회차만 검색")
        print(f"📚 관련 문서 검색 중...")
        
        # 검색 전략 단순화: 1번만 검색 (속도 개선)
        documents = self.retriever.search_similar_documents(
            question,
            limit=top_k,
            session_name=session_name,
            threshold=0.3,  # 적절한 임계값
        )
        
        # session_name이 있는 경우 검증 및 필터링
        if session_name and documents:
            verified_docs = [
                doc for doc in documents
                if doc.get("metadata", {}).get("session_name") == session_name
            ]
            if verified_docs:
                documents = verified_docs[:top_k]
            else:
                # 필터링 실패 시 관련 문서라도 사용 (하지만 경고)
                print(f"⚠️  {session_name} 회차 문서를 찾지 못했지만, 관련 문서 {len(documents)}개를 사용합니다.")
        
        final_documents = documents
        
        if not final_documents:
            # 관련 정보가 전혀 없는 경우
            return {
                "answer": (
                    f"죄송합니다. '{question}'에 대한 관련 정보를 찾을 수 없습니다. "
                    f"{'특정 회차(' + session_name + ')의 ' if session_name else ''}"
                    "다른 질문이나 키워드로 시도해보시거나, 전체 데이터에서 검색해보시기 바랍니다."
                ),
                "sources": [],
                "question": question,
            }
        
        documents = final_documents
        
        print(f"✅ {len(documents)}개 관련 문서 발견")
        
        # 2. 컨텍스트 구성 (검색된 문서 사용)
        context_parts = []
        session_mismatch_count = 0  # 필터링 불일치 카운트
        
        for idx, doc in enumerate(documents, start=1):
            metadata = doc.get("metadata", {})
            source_type = doc.get("source_type", "")
            doc_session = metadata.get("session_name", "")
            
            # 필터링 불일치 확인
            if session_name and doc_session and doc_session != session_name:
                session_mismatch_count += 1
            
            header = f"[문서 {idx}]"
            if source_type == "session_summary":
                header += f" 세션 요약: {doc_session}"
            elif source_type == "party_position":
                header += f" 정당 입장: {metadata.get('party_name', '')} - {metadata.get('agenda_title', '')}"
                if doc_session:
                    header += f" (회차: {doc_session})"
            elif source_type == "qa_pair":
                header += f" 질의응답: {metadata.get('questioner', '')} → {metadata.get('respondent', '')}"
                if doc_session:
                    header += f" (회차: {doc_session})"
            elif source_type == "agenda_analysis":
                header += f" 안건 분석: {metadata.get('agenda_title', '')}"
                if doc_session:
                    header += f" (회차: {doc_session})"
            
            context_parts.append(f"{header}\n{doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        # 필터링 불일치가 있으면 컨텍스트에 명시
        context_note = ""
        if session_name and session_mismatch_count > 0:
            context_note = f"\n\n주의: 요청하신 {session_name} 회차로 필터링했지만, 일부 문서는 다른 회차의 정보일 수 있습니다."
        
        # 3. LLM에 질문 + 컨텍스트 전달하여 답변 생성
        prompt = self._build_qa_prompt(question, context, session_name=session_name, context_note=context_note)
        
        print(f"🤖 LLM으로 답변 생성 중...")
        
        def _call_llm():
            return self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "국회 회의록 분석 전문가입니다. "
                            "반드시 높임말을 사용하여 답변하세요. "
                            "제공된 컨텍스트를 기반으로 핵심만 간결하게 답변합니다. "
                            "불필요한 상세 설명이나 반복은 피하고, 핵심 정보만 전달하세요. "
                            "컨텍스트에 없는 내용은 추측하지 마세요."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=800,  # 속도 개선: 토큰 수 제한 (기본값보다 낮게 설정)
            )
        
        response = self._retry_with_backoff(_call_llm)
        answer = response.choices[0].message.content
        
        result = {
            "answer": answer,
            "question": question,
        }
        
        if include_sources:
            result["sources"] = [
                {
                    "content": doc["content"][:200] + "...",  # 일부만 표시
                    "metadata": doc["metadata"],
                    "similarity": doc["similarity"],
                    "source_type": doc["source_type"],
                }
                for doc in documents
            ]
        
        return result

    def _build_qa_prompt(
        self, 
        question: str, 
        context: str, 
        session_name: Optional[str] = None,
        context_note: str = ""
    ) -> str:
        """RAG 프롬프트 구성"""
        session_filter_note = ""
        if session_name:
            session_filter_note = f"\n참고: 이 질문은 '{session_name}' 회차에 대한 질문입니다. 가능한 한 해당 회차의 정보를 우선적으로 사용하되, 관련 정보가 부족하면 다른 회차의 정보도 참고할 수 있습니다."
        
        return f"""국회 회의록 분석 전문가입니다. 다음은 검색된 관련 문서입니다:

{context}{context_note}{session_filter_note}

위 문서를 참고하여 다음 질문에 높임말로 답변해주세요:

질문: {question}

답변 지침:
1. 반드시 높임말 사용 (반말 금지)
2. 첫 문장에 핵심 답변 명확히 제시
3. 간결하고 요점만 전달 (불필요한 반복 제거)
4. 핵심만 요약 (의원 발언 나열 지양)
5. 문서에서 확인된 사실만 언급 (추측 금지)
6. 정보 부족 시 관련 정보라도 제공

답변 형식:
- 첫 문장: 핵심 답변
- 이후: 핵심 요약 (최대 2-3문장)
- 높임말 사용 필수
"""

    def ask_multiple_questions(
        self,
        questions: List[str],
        *,
        session_name: Optional[str] = None,
    ) -> List[Dict[str, any]]:
        """여러 질문에 대한 답변 생성"""
        results = []
        for question in questions:
            result = self.ask_question(question, session_name=session_name)
            results.append(result)
        return results
