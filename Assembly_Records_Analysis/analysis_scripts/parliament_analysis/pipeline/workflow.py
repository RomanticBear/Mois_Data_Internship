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
        """간단한 휴리스틱으로 노이즈 발언을 제거."""
        def is_valid_speech(row: pd.Series) -> bool:
            text = str(row.get("speech_text", ""))
            if pd.isna(text) or not text.strip():
                return False
            if len(text.strip()) < 30:
                return False
            return True

        quality_mask = df.apply(is_valid_speech, axis=1)
        return df[quality_mask].copy()

    @staticmethod
    def compute_dataframe_hash(df: pd.DataFrame) -> str:
        """Generate a deterministic hash for a session dataframe."""
        json_payload = df.sort_index(axis=1).to_json(orient="records", force_ascii=False)
        return hashlib.sha256(json_payload.encode("utf-8")).hexdigest()

    def prepare_session_summary_payload(self, df: pd.DataFrame) -> Dict[str, Any]:
        """회차 요약을 위한 데이터 준비."""
        agenda_stats: Dict[str, int] = {}
        for agenda in df["agenda_item_titles"].dropna().unique():
            if pd.notna(agenda) and str(agenda).strip():
                agenda_count = len(df[df["agenda_item_titles"] == agenda])
                agenda_stats[str(agenda)] = agenda_count

        party_stats = df["party"].value_counts().to_dict()
        speeches_sample: List[Dict[str, Any]] = []

        parties = df["party"].dropna().unique()
        for party in parties[:5]:
            party_speeches = df[df["party"] == party]
            if len(party_speeches) > 0:
                sample_size = min(5, len(party_speeches))
                sample_speeches = party_speeches.sample(n=sample_size, random_state=42)
                for _, row in sample_speeches.iterrows():
                    speeches_sample.append(
                        {
                            "party": party,
                            "speaker": row.get("speaker_name", ""),
                            "text": str(row.get("speech_text", ""))[:300],
                        }
                    )

        return {
            "total_speeches": len(df),
            "agenda_stats": agenda_stats,
            "party_stats": party_stats,
            "speeches_sample": speeches_sample[:20],
        }

    def prepare_agenda_payloads(
        self, df: pd.DataFrame, *, top_agendas: int = 3
    ) -> List[Dict[str, Any]]:
        """안건별 발언을 LLM 프롬프트에 맞게 정리."""
        agenda_counts = df["agenda_item_titles"].value_counts()
        top_agenda_titles = [
            title for title in agenda_counts.head(top_agendas).index.tolist() if pd.notna(title)
        ]

        payloads: List[Dict[str, Any]] = []
        for agenda_title in top_agenda_titles:
            if not str(agenda_title).strip():
                continue

            agenda_df = df[df["agenda_item_titles"] == agenda_title]
            party_speeches: Dict[str, List[str]] = {}
            for party in agenda_df["party"].dropna().unique():
                party_data = agenda_df[agenda_df["party"] == party]
                sample_size = min(5, len(party_data))
                samples = party_data.sample(n=sample_size, random_state=42)
                party_speeches[str(party)] = [
                    str(row.get("speech_text", ""))[:400] for _, row in samples.iterrows()
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


