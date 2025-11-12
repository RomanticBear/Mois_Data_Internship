"""
[역할] 파이프라인 유틸리티 함수
- generate_analysis_version(): 분석 버전 생성 (타임스탬프 기반)
- PipelineContext: 파이프라인 컨텍스트 정보 (실행자, 실행 시간 등)
- 공통 유틸리티 함수 제공
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class PipelineContext:
    """Runtime context information shared across pipeline steps."""

    initiated_by: str
    triggered_at: datetime = datetime.now(timezone.utc)
    notes: Optional[str] = None


def generate_analysis_version() -> str:
    """Produce a semantic version or timestamp-based revision string."""
    return datetime.now(timezone.utc).strftime("analysis-%Y%m%d%H%M%S")


