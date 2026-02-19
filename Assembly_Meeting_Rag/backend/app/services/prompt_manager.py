"""
프롬프트 관리 서비스
질문 유형별 프롬프트 관리
"""
from enum import Enum


class QuestionType(str, Enum):
    """질문 유형"""
    SUMMARY = "summary"  # 회의 요약
    ISSUE = "issue"  # 쟁점 정리
    SPEAKER = "speaker"  # 발언자별 발언 정리
    MATERIAL_REQUEST = "material_request"  # 자료제출요구 목록
    NEXT_MEETING_PREP = "next_meeting_prep"  # 다음 회의 준비 포인트
    GENERAL = "general"  # 일반 질문


class PromptManager:
    """프롬프트 관리"""
    BASE_INSTRUCTIONS = """당신은 국회 행정안전위원회 회의록 분석 보좌관입니다.
답변은 항상 한국어로 작성합니다.

[핵심 원칙]
- 회의록 근거 중심으로 답하고, 추정은 금지합니다.
- 회차/날짜/인물/기관/법안명은 문서에 있는 표현을 우선 사용합니다.
- 근거가 약하거나 확인되지 않으면 "문서 내 근거 없음"을 명확히 씁니다.
- 질문 의도에 맞는 길이와 형식으로 답합니다. (짧게/분석형/비교형)

[표현 원칙]
- 답변은 자연스럽고 읽기 쉬운 문장으로 작성합니다.
- 기계적인 고정 템플릿(예: "1) 한 줄 결론:")은 사용하지 않습니다.
- 필요한 경우에만 소제목/불릿을 사용하고, 과도한 형식화는 피합니다.
"""

    PROMPTS = {
        QuestionType.SUMMARY: """요약 질문입니다.
핵심 안건, 주요 쟁점, 결정사항과 남은 과제를 균형 있게 정리하세요.""",

        QuestionType.ISSUE: """쟁점 분석 질문입니다.
중요 쟁점을 우선순위대로 정리하고, 쟁점별 상태(논의/합의/보류)와 리스크를 설명하세요.""",

        QuestionType.SPEAKER: """발언자 분석 질문입니다.
주요 발언자 중심으로 핵심 주장과 차이점을 정리하고, 발언의 맥락을 간단히 설명하세요.""",

        QuestionType.MATERIAL_REQUEST: """자료요구 질문입니다.
요청 주체, 요청 자료, 대상 기관, 제출 상태를 중심으로 정리하세요.""",

        QuestionType.NEXT_MEETING_PREP: """다음 회의 준비 질문입니다.
이어질 가능성이 높은 쟁점과 사전 준비 포인트를 실무 관점으로 제시하세요.""",

        QuestionType.GENERAL: """일반 질문입니다.
질문 의도를 먼저 파악한 뒤, 필요한 수준으로 정확하고 자연스럽게 답변하세요."""
    }
    
    def get_prompt(self, question_type: QuestionType, user_question: str) -> str:
        """
        질문 유형에 맞는 프롬프트 생성
        
        Args:
            question_type: 질문 유형
            user_question: 사용자 질문
            
        Returns:
            완성된 프롬프트
        """
        base = self.BASE_INSTRUCTIONS
        type_prompt = self.PROMPTS.get(question_type, self.PROMPTS[QuestionType.GENERAL])
        
        return f"""{base}

질문 유형 가이드:
{type_prompt}

사용자 질문:
{user_question}

응답 지시:
- 질문이 단순 사실 확인이면 3~5문장 내로 간결하게 답합니다.
- 질문이 비교/분석형이면 핵심 근거를 묶어서 설명합니다.
- 질문 범위가 넓으면 먼저 범위를 짧게 정의한 뒤 답합니다.
- 마지막 문장은 불필요한 반복 없이 깔끔하게 마무리합니다.
"""
    
    def classify_question(self, question: str) -> QuestionType:
        """
        질문 유형 자동 분류 (간단한 키워드 기반)
        실제로는 더 정교한 분류기 필요
        
        Args:
            question: 사용자 질문
            
        Returns:
            질문 유형
        """
        question_lower = question.lower()
        
        # 키워드 기반 분류
        if any(keyword in question_lower for keyword in ["요약", "개요", "주요 내용", "개관", "정리해", "핵심만"]):
            return QuestionType.SUMMARY
        
        if any(keyword in question_lower for keyword in ["쟁점", "논의", "이슈", "문제", "갈등", "충돌", "논란"]):
            return QuestionType.ISSUE
        
        if any(keyword in question_lower for keyword in ["발언", "의견", "발언자", "누가", "입장"]):
            return QuestionType.SPEAKER
        
        if any(keyword in question_lower for keyword in ["자료", "제출", "요구", "요청", "보고", "근거자료"]):
            return QuestionType.MATERIAL_REQUEST
        
        if any(keyword in question_lower for keyword in ["다음 회의", "준비", "후속", "이어서", "최신", "최근", "가장 최근", "향후"]):
            return QuestionType.NEXT_MEETING_PREP
        
        return QuestionType.GENERAL





