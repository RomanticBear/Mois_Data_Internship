"""
프롬프트 관리 서비스
질문 유형별 프롬프트 관리
"""
from enum import Enum
from typing import Optional


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
    
    BASE_INSTRUCTIONS = """당신은 국회회의록을 기반으로 질문에 답변하는 전문 AI입니다.
다음 원칙을 따라야 합니다:
1. 반드시 문서 내 근거를 기반으로 답변하세요.
2. 가능하면 페이지나 발언 맥락을 명시하세요.
3. 불확실한 경우 "문서 내 근거 없음"을 명시하세요.
4. 답변은 객관적이고 사실에 기반해야 합니다.
"""
    
    PROMPTS = {
        QuestionType.SUMMARY: """회의록을 기반으로 다음 회의의 주요 내용을 요약해주세요:
- 주요 안건
- 핵심 쟁점
- 중요한 결정사항
- 다음 회의로 이어질 이슈""",
        
        QuestionType.ISSUE: """최근 회의에서 논의된 주요 쟁점들을 정리해주세요:
- 각 쟁점의 핵심 내용
- 논의 과정
- 결론 또는 향후 과제""",
        
        QuestionType.SPEAKER: """발언자별로 주요 발언 내용을 정리해주세요:
- 발언자 이름
- 주요 발언 내용
- 핵심 의견""",
        
        QuestionType.MATERIAL_REQUEST: """회의 중 요구된 자료제출요구 사항을 정리해주세요:
- 요구한 발언자
- 요구한 자료
- 제출 기한 (있다면)
- 담당 기관""",
        
        QuestionType.NEXT_MEETING_PREP: """다음 회의를 준비하기 위한 포인트를 정리해주세요:
- 이어서 논의될 가능성이 있는 쟁점
- 아직 해결되지 않은 이슈
- 추가로 준비해야 할 자료
- 확인이 필요한 사항""",
        
        QuestionType.GENERAL: """사용자의 질문에 대해 회의록을 기반으로 답변해주세요."""
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

{type_prompt}

사용자 질문: {user_question}

위 질문에 대해 문서를 기반으로 상세히 답변해주세요."""
    
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
        if any(keyword in question_lower for keyword in ["요약", "개요", "주요 내용", "개관"]):
            return QuestionType.SUMMARY
        
        if any(keyword in question_lower for keyword in ["쟁점", "논의", "이슈", "문제"]):
            return QuestionType.ISSUE
        
        if any(keyword in question_lower for keyword in ["발언", "의견", "발언자"]):
            return QuestionType.SPEAKER
        
        if any(keyword in question_lower for keyword in ["자료", "제출", "요구", "요청"]):
            return QuestionType.MATERIAL_REQUEST
        
        if any(keyword in question_lower for keyword in ["다음 회의", "준비", "후속", "이어서"]):
            return QuestionType.NEXT_MEETING_PREP
        
        return QuestionType.GENERAL





