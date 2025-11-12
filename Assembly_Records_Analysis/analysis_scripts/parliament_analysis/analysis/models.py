"""
[역할] 분석 결과 데이터 클래스 정의
- SessionSummary: 세션 요약 정보
- PartyPosition: 정당별 입장 정보
- AgendaPartyAnalysis: 안건별 정당 분석 결과
- QAAnalysisMetrics: QA 분석 지표
- IssueTrend: 이슈 트렌드 정보
- JSON 직렬화 가능한 dataclass로 정의되어 분석, 저장, RAG 레이어 간 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class SessionSummary:
    """High-level summary of a plenary session."""

    session_name: str
    meeting_date: datetime | None
    key_issues: List[Dict[str, Any]]
    overall_sentiment: Optional[float] = None
    raw_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PartyPosition:
    """Aggregated stance information for a party on a specific agenda item."""

    session_name: str
    agenda_title: str
    party_name: str
    stance_label: str
    key_points: List[str]
    concerns: List[str]
    suggestions: List[str]
    summary_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgendaPartyAnalysis:
    """Bundle of party positions and meta insights for a single agenda."""

    session_name: str
    agenda_title: str
    party_positions: List[PartyPosition]
    consensus_points: List[str]
    conflict_points: List[str]
    cooperation_level: Optional[str]
    summary_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QAInteraction:
    """Question-answer pair effectiveness metrics."""

    session_name: str
    agenda_title: str
    questioner: str
    respondent: str
    question_text: str
    answer_text: str
    effectiveness_score: float
    effectiveness_bucket: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QAAnalysisMetrics:
    """Aggregated QA effectiveness results for a session."""

    session_name: str
    total_qa_pairs: int
    quality_distribution: Dict[str, Any]
    question_types: Dict[str, Any]
    answer_quality: Dict[str, Any]
    key_issues: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IssueTrend:
    """Issue-level sentiment/frequency tracking per session."""

    session_name: str
    issue_keyword: str
    sentiment_score: Optional[float]
    frequency: int
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


