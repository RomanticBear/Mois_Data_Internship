"""
문서 메타데이터 모델
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class DocumentMetadata(BaseModel):
    """문서 메타데이터"""
    id: Optional[int] = None
    filename: str
    assembly_number: str  # 국회 회차 (예: "제415회")
    session_type: str  # 회기 유형 (예: "임시회")
    committee: str  # 위원회 (예: "행정안전위원회")
    meeting_number: int  # 회의 번호 (예: 1차, 2차)
    date: str  # 날짜 (예: "2024.06.13")
    vector_store_file_id: Optional[str] = None  # OpenAI Vector Store 파일 ID
    is_active: bool = True  # Active Window에 포함 여부
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentUploadRequest(BaseModel):
    """문서 업로드 요청"""
    assembly_number: str
    session_type: str
    committee: str
    meeting_number: int
    date: str


class DocumentResponse(BaseModel):
    """문서 응답"""
    id: int
    filename: str
    assembly_number: str
    session_type: str
    committee: str
    meeting_number: int
    date: str
    is_active: bool
    created_at: datetime


class QueryRequest(BaseModel):
    """질문 요청"""
    question: str
    question_type: Optional[str] = None  # 질문 유형 (자동 분류 또는 명시)
    include_inactive: bool = False  # 비활성 문서 포함 여부


class QueryResponse(BaseModel):
    """질문 응답"""
    answer: str
    sources: list[dict]  # 근거 문서 및 스니펫
    question_type: str
    metadata: Optional[dict] = None





