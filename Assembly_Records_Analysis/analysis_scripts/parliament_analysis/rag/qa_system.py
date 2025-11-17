"""
[역할] RAG 기반 질문-답변 시스템
- ask_question(): 질문을 받아서 벡터 DB에서 관련 문서를 검색하고 답변 생성
- RAGRetriever를 사용하여 관련 문서 검색
- OpenAI LLM을 사용하여 컨텍스트 기반 답변 생성
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

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
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature

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
        # 1. 관련 문서 검색
        print(f"🔍 질문: {question}")
        if session_name:
            print(f"📌 필터링: {session_name} 회차만 검색")
        print(f"📚 관련 문서 검색 중...")
        
        # session_name 없이 먼저 검색해보기 (디버깅)
        documents = self.retriever.search_similar_documents(
            question,
            limit=top_k * 2,  # 더 많이 가져오기
            session_name=None,  # 일단 필터링 없이 검색
            threshold=0.3,  # 낮은 임계값
        )
        
        # session_name 필터링 적용
        if session_name and documents:
            filtered_docs = [
                doc for doc in documents
                if doc.get("metadata", {}).get("session_name") == session_name
            ]
            if filtered_docs:
                documents = filtered_docs[:top_k]
            else:
                print(f"⚠️  {session_name} 회차 문서를 찾을 수 없습니다. 전체 검색 결과 사용")
                documents = documents[:top_k]
        
        if not documents:
            # 임계값을 더 낮춰서 재시도
            documents = self.retriever.search_similar_documents(
                question,
                limit=top_k,
                session_name=None,
                threshold=0.1,
            )
        
        if not documents:
            return {
                "answer": "죄송합니다. 관련된 정보를 찾을 수 없습니다.",
                "sources": [],
                "question": question,
            }
        
        print(f"✅ {len(documents)}개 관련 문서 발견")
        
        # 2. 컨텍스트 구성 (검색된 문서 사용)
        context_parts = []
        for idx, doc in enumerate(documents, start=1):
            metadata = doc.get("metadata", {})
            source_type = doc.get("source_type", "")
            
            header = f"[문서 {idx}]"
            if source_type == "session_summary":
                header += f" 세션 요약: {metadata.get('session_name', '')}"
            elif source_type == "party_position":
                header += f" 정당 입장: {metadata.get('party_name', '')} - {metadata.get('agenda_title', '')}"
            elif source_type == "qa_pair":
                header += f" 질의응답: {metadata.get('questioner', '')} → {metadata.get('respondent', '')}"
            
            context_parts.append(f"{header}\n{doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        # 3. LLM에 질문 + 컨텍스트 전달하여 답변 생성
        prompt = self._build_qa_prompt(question, context)
        
        print(f"🤖 LLM으로 답변 생성 중...")
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 국회 회의록 분석 전문가입니다. "
                        "제공된 컨텍스트를 기반으로 정확하고 구체적인 답변을 제공합니다. "
                        "컨텍스트에 없는 내용은 추측하지 마세요."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        
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

    def _build_qa_prompt(self, question: str, context: str) -> str:
        """RAG 프롬프트 구성"""
        return f"""당신은 국회 회의록 분석 전문가입니다. 다음은 국회 회의록에서 검색된 관련 문서입니다:

{context}

위 문서들을 참고하여 다음 질문에 답변해주세요:

질문: {question}

답변 시 다음 사항을 지켜주세요:
1. 제공된 컨텍스트를 기반으로 정확하고 구체적으로 답변
2. 문서에서 언급된 구체적인 정보를 포함 (회차, 안건, 정당, 발언자 등)
3. 여러 문서의 정보를 종합하여 답변
4. 컨텍스트에 명확히 나와있는 내용은 반드시 포함
5. 한국어로 자연스럽고 상세하게 답변
6. 컨텍스트에 정보가 있으면 반드시 답변하고, 정말 없는 경우에만 "정보를 찾을 수 없습니다"라고 답변
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

