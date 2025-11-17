"""
[역할] 데이터 준비와 분석 실행 워크플로우 조율
- load_session_data(): 세션 데이터 로드
- filter_quality_speeches(): 품질 발언 필터링
- prepare_session_summary_payload(): 세션 요약 데이터 준비
- prepare_agenda_payloads(): 안건별 데이터 준비
- prepare_qa_pairs(): QA 페어 추출
- run_session_summary(): 세션 요약 실행
- run_party_positions(): 정당 입장 분석 실행
- run_qa_analysis(): QA 분석 실행
- 데이터 전처리와 LLM 분석 실행을 연결하는 오케스트레이터
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ..analysis.models import AgendaPartyAnalysis, QAAnalysisMetrics, SessionSummary
from ..analysis.openai_analyzer import OpenAISessionAnalyzer


class SessionAnalysisWorkflow:
    """데이터 준비와 LLM 분석을 연결하는 워크플로."""

    def __init__(self, *, openai_client, model: str = "gpt-4o-mini", temperature: float = 0.3) -> None:
        self.analyzer = OpenAISessionAnalyzer(
            llm_client=openai_client,
            model=model,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # 데이터 준비
    # ------------------------------------------------------------------

    @staticmethod
    def load_session_data(
        session_name: str,
        *,
        data_root: Optional[Path] = None,
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """Load a session dataframe from the project data directory."""
        if data_root is None:
            current_dir = Path(__file__).resolve().parents[3]  # analysis_scripts/
            data_root = current_dir / "data" / "with_party"

        session_dir = data_root / session_name
        if not session_dir.exists():
            raise FileNotFoundError(f"{session_dir} 디렉토리를 찾을 수 없습니다.")

        speech_files = sorted(
            file for file in session_dir.iterdir() if file.name.endswith(".csv") and "speeches" in file.name
        )
        if not speech_files:
            raise FileNotFoundError(f"{session_dir} 안에서 발언 CSV 파일을 찾을 수 없습니다.")

        frames: List[pd.DataFrame] = []
        for file in speech_files:
            df = pd.read_csv(file, encoding=encoding)
            df["session_name"] = session_name
            df["file_name"] = file.name
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def filter_quality_speeches(df: pd.DataFrame) -> pd.DataFrame:
        """강화된 품질 필터링: 의사진행 발언 및 노이즈 제거."""
        # 의사진행 발언 패턴
        procedural_patterns = [
            "의석을 정돈", "회의를 개최", "감사의 말씀", "다음은", "상정합니다",
            "개회를 선포", "폐회를 선포", "의원 여러분", "좋은 말씀", "참석해 주셔서",
            "다음 안건", "이상으로", "다음 순서", "의사진행", "회의 진행",
            "감사합니다", "수고하셨습니다", "이상입니다", "다음으로 넘어가",
            "의석 정돈", "회의 시작", "회의 종료"
        ]
        
        def is_valid_speech(row: pd.Series) -> bool:
            text = str(row.get("speech_text", "")).strip()
            
            # 기본 검증
            if pd.isna(text) or not text:
                return False
            if len(text) < 50:  # 최소 길이 증가 (30 -> 50)
                return False
            
            # 의사진행 발언 제외
            text_lower = text.lower()
            if any(pattern in text_lower for pattern in procedural_patterns):
                # 패턴이 포함되어 있지만 실제 내용이 충분히 길면 포함
                if len(text) < 100:
                    return False
            
            # 반복 문자 패턴 제외 (예: "감사합니다" 반복)
            if len(text) > 0 and len(set(text[:min(20, len(text))])) < 5:
                return False
            
            return True

        quality_mask = df.apply(is_valid_speech, axis=1)
        return df[quality_mask].copy()

    @staticmethod
    def compute_dataframe_hash(df: pd.DataFrame) -> str:
        """Generate a deterministic hash for a session dataframe."""
        json_payload = df.sort_index(axis=1).to_json(orient="records", force_ascii=False)
        return hashlib.sha256(json_payload.encode("utf-8")).hexdigest()

    @staticmethod
    def calculate_importance_score(row: pd.Series) -> float:
        """발언의 중요도 점수 계산 (0-1)."""
        text = str(row.get("speech_text", "")).strip()
        text_lower = text.lower()
        
        score = 0.0
        
        # 1. 길이 점수 (적당한 길이가 중요)
        length = len(text)
        if 100 <= length <= 1000:
            score += 0.2
        elif 50 <= length < 100 or 1000 < length <= 2000:
            score += 0.1
        
        # 2. 정책 키워드 점수
        policy_keywords = [
            "정책", "법안", "예산", "제안", "건의", "개선", "문제", "해결",
            "국민", "사회", "경제", "복지", "교육", "보건", "환경", "안전",
            "비판", "지적", "우려", "필요", "중요", "시급", "개선안", "대안",
            "정부", "국회", "입법", "법률", "규정", "제도", "시스템"
        ]
        keyword_count = sum(1 for keyword in policy_keywords if keyword in text_lower)
        score += min(keyword_count * 0.08, 0.4)  # 최대 0.4점
        
        # 3. 질문/답변 패턴 점수
        qa_patterns = ["질의", "질문", "답변", "설명", "문의", "알고 싶", "묻고 싶"]
        if any(pattern in text_lower for pattern in qa_patterns):
            score += 0.2
        
        # 4. 숫자/통계 포함 점수 (구체적인 내용)
        if re.search(r'\d+[억만천]', text) or re.search(r'\d+%', text) or re.search(r'\d+원', text):
            score += 0.15
        
        # 5. 문장 복잡도 점수 (단순 반복이 아닌 실제 내용)
        sentences = text.split('。') + text.split('.') + text.split('!') + text.split('?') + text.split('\n')
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 20]
        if len(meaningful_sentences) >= 3:
            score += 0.1
        elif len(meaningful_sentences) >= 2:
            score += 0.05
        
        # 6. 의사진행 발언 패턴 감지 (점수 감점)
        procedural_patterns = [
            "의석을 정돈", "회의를 개최", "감사의 말씀", "다음은", "상정합니다"
        ]
        if any(pattern in text_lower for pattern in procedural_patterns) and len(text) < 100:
            score *= 0.3  # 의사진행 발언은 점수 대폭 감소
        
        return min(score, 1.0)  # 최대 1.0

    @staticmethod
    def select_important_speeches(
        df: pd.DataFrame,
        *,
        max_speeches: int = 50,
        min_importance: float = 0.3,
        party_balance: bool = True
    ) -> pd.DataFrame:
        """중요도 기반으로 발언 선별."""
        
        # 중요도 점수 계산
        df = df.copy()
        df['importance_score'] = df.apply(
            lambda row: SessionAnalysisWorkflow.calculate_importance_score(row),
            axis=1
        )
        
        # 최소 중요도 이상만 선택
        filtered_df = df[df['importance_score'] >= min_importance].copy()
        
        if len(filtered_df) == 0:
            # 중요도 기준을 만족하는 발언이 없으면 점수 상위 N개 선택
            df = df.sort_values('importance_score', ascending=False)
            return df.head(max_speeches)
        
        df = filtered_df
        
        if party_balance:
            # 정당별 균형 유지하면서 중요도 높은 것 선택
            selected_indices = set()
            parties = df['party'].dropna().unique()
            if len(parties) > 0:
                speeches_per_party = max(1, max_speeches // len(parties))
                
                for party in parties:
                    party_speeches = df[df['party'] == party].copy()
                    party_speeches = party_speeches.sort_values(
                        'importance_score', ascending=False
                    )
                    # 아직 선택되지 않은 발언 중에서 선택
                    party_available = party_speeches[~party_speeches.index.isin(selected_indices)]
                    selected = party_available.head(speeches_per_party)
                    selected_indices.update(selected.index.tolist())
            
            # 정당별 균형 후 남은 자리 채우기
            remaining = max_speeches - len(selected_indices)
            if remaining > 0:
                all_available = df[~df.index.isin(selected_indices)].copy()
                all_available = all_available.sort_values(
                    'importance_score', ascending=False
                )
                additional_indices = all_available.head(remaining).index.tolist()
                selected_indices.update(additional_indices)
            
            result = df[df.index.isin(selected_indices)].copy()
            result = result.sort_values('importance_score', ascending=False)
            return result.head(max_speeches)
        else:
            # 단순 중요도 순 정렬
            df = df.sort_values('importance_score', ascending=False)
            return df.head(max_speeches)

    def prepare_session_summary_payload(
        self, 
        df: pd.DataFrame,
        *,
        max_sample_speeches: int = 50,
        max_chars_per_speech: int = 500
    ) -> Dict[str, Any]:
        """회차 요약을 위한 데이터 준비 (중요도 기반 선별)."""
        # 통계 계산
        agenda_stats: Dict[str, int] = {}
        for agenda in df["agenda_item_titles"].dropna().unique():
            if pd.notna(agenda) and str(agenda).strip():
                agenda_count = len(df[df["agenda_item_titles"] == agenda])
                agenda_stats[str(agenda)] = agenda_count

        party_stats = df["party"].value_counts().to_dict()
        
        # 중요도 기반 선별
        important_speeches = self.select_important_speeches(
            df,
            max_speeches=max_sample_speeches,
            min_importance=0.3,
            party_balance=True
        )
        
        # 샘플 데이터 구성
        speeches_sample: List[Dict[str, Any]] = []
        for _, row in important_speeches.iterrows():
            speeches_sample.append({
                "party": row.get("party", ""),
                "speaker": row.get("speaker_name", ""),
                "text": str(row.get("speech_text", ""))[:max_chars_per_speech],
            })

        return {
            "total_speeches": len(df),
            "quality_speeches": len(important_speeches),
            "agenda_stats": agenda_stats,
            "party_stats": party_stats,
            "speeches_sample": speeches_sample,
        }

    def prepare_agenda_payloads(
        self, 
        df: pd.DataFrame, 
        *,
        top_agendas: int = 3,
        max_speeches_per_party: int = 10
    ) -> List[Dict[str, Any]]:
        """안건별 발언을 LLM 프롬프트에 맞게 정리 (중요도 기반)."""
        agenda_counts = df["agenda_item_titles"].value_counts()
        top_agenda_titles = [
            title for title in agenda_counts.head(top_agendas).index.tolist() if pd.notna(title)
        ]

        payloads: List[Dict[str, Any]] = []
        for agenda_title in top_agenda_titles:
            if not str(agenda_title).strip():
                continue

            agenda_df = df[df["agenda_item_titles"] == agenda_title].copy()
            
            # 중요도 점수 계산
            agenda_df['importance_score'] = agenda_df.apply(
                lambda row: self.calculate_importance_score(row),
                axis=1
            )
            
            party_speeches: Dict[str, List[str]] = {}
            for party in agenda_df["party"].dropna().unique():
                party_data = agenda_df[agenda_df["party"] == party].copy()
                # 중요도 순 정렬
                party_data = party_data.sort_values(
                    'importance_score', ascending=False
                )
                # 상위 N개 선택
                selected = party_data.head(max_speeches_per_party)
                party_speeches[str(party)] = [
                    str(row.get("speech_text", ""))[:500] 
                    for _, row in selected.iterrows()
                ]

            payloads.append(
                {
                    "agenda_title": str(agenda_title),
                    "total_speeches": len(agenda_df),
                    "party_speeches": party_speeches,
                }
            )
        return payloads

    def prepare_qa_pairs(self, df: pd.DataFrame, session_name: str) -> List[Dict[str, str]]:
        """질의-응답 쌍 추출."""
        qa_pairs: List[Dict[str, str]] = []
        speeches_list = df.to_dict("records")

        question_markers = ["질의", "질문", "?", "문의", "묻고 싶", "알고 싶"]
        answer_markers = ["답변", "설명", "말씀", "드리", "알려"]

        for i in range(len(speeches_list) - 1):
            curr = speeches_list[i]
            next_sp = speeches_list[i + 1]

            curr_text = str(curr.get("speech_text", "")).lower()
            next_text = str(next_sp.get("speech_text", "")).lower()

            is_question = any(marker in curr_text for marker in question_markers) or "?" in curr_text
            is_answer = any(marker in next_text for marker in answer_markers)

            if is_question and is_answer:
                qa_pairs.append(
                    {
                        "session_name": session_name,
                        "question": str(curr.get("speech_text", ""))[:500],
                        "questioner": curr.get("speaker_name", ""),
                        "question_party": curr.get("party", ""),
                        "answer": str(next_sp.get("speech_text", ""))[:500],
                        "answerer": next_sp.get("speaker_name", ""),
                        "answer_party": next_sp.get("party", ""),
                    }
                )

        return qa_pairs

    # ------------------------------------------------------------------
    # 분석 실행
    # ------------------------------------------------------------------

    def run_session_summary(
        self, session_name: str, *, payload: Dict[str, Any]
    ) -> Optional[SessionSummary]:
        return self.analyzer.analyze_session_summary(
            session_name=session_name,
            session_payload=payload,
        )

    def run_party_positions(
        self, session_name: str, *, agenda_payloads: Sequence[Dict[str, Any]]
    ) -> List[AgendaPartyAnalysis]:
        analyses: List[AgendaPartyAnalysis] = []
        if not agenda_payloads:
            return analyses

        responses = self.analyzer.analyze_party_positions(
            session_name=session_name,
            agenda_payloads=agenda_payloads,
        )
        analyses.extend(responses)
        return analyses

    def run_qa_analysis(
        self,
        session_name: str,
        *,
        qa_pairs: Sequence[Dict[str, str]],
    ) -> Optional[QAAnalysisMetrics]:
        if not qa_pairs:
            return None
        return self.analyzer.analyze_qa_effectiveness(qa_pairs=qa_pairs)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def asdict_or_none(obj: Optional[Any]) -> Optional[Dict[str, Any]]:
        if obj is None:
            return None
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return obj


