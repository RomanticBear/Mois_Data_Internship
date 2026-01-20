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
            data_root = current_dir / "data"  # data/제oo회/ 구조로 변경

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
        max_speeches: Optional[int] = None,
        sample_ratio: float = 0.8,
        min_samples: int = 50,
        max_samples: Optional[int] = None,
        min_importance: Optional[float] = None,
        skip_importance_filter: bool = True,
        party_balance: bool = True
    ) -> pd.DataFrame:
        """발언 선별 (중요도 필터링 옵션 포함).
        
        Parameters
        ----------
        df : pd.DataFrame
            필터링할 발언 데이터
        max_speeches : Optional[int], optional
            최대 선택할 발언 개수 (고정값). None이면 sample_ratio 사용
        sample_ratio : float, default 0.8
            전체 발언 대비 샘플 비율 (0.0 ~ 1.0). max_speeches가 None일 때 사용
            기본값 80%로 설정하여 대부분의 데이터 포함
        min_samples : int, default 50
            최소 선택할 발언 개수 (비율 기반 계산 시)
        max_samples : Optional[int], optional
            최대 선택할 발언 개수 (비율 기반 계산 시). None이면 제한 없음
        min_importance : Optional[float], optional
            최소 중요도 점수 (0.0 ~ 1.0). None이면 중요도 필터링 안 함
        skip_importance_filter : bool, default True
            True면 중요도 필터링을 완전히 건너뛰고 비율만으로 선별
            False면 min_importance 기준으로 필터링
        party_balance : bool, default True
            정당별 균형 유지 여부
        """
        # max_speeches 계산: 비율 기반 또는 고정값
        if max_speeches is None:
            # 비율 기반 계산 (최소/최대 제한 적용)
            calculated_max = int(len(df) * sample_ratio)
            if max_samples is not None:
                max_speeches = max(min_samples, min(calculated_max, max_samples))
            else:
                max_speeches = max(min_samples, calculated_max)
        
        # 중요도 점수 계산 (참고용, 필터링에는 사용 안 함)
        df = df.copy()
        df['importance_score'] = df.apply(
            lambda row: SessionAnalysisWorkflow.calculate_importance_score(row),
            axis=1
        )
        
        # 중요도 필터링 건너뛰기 옵션
        if skip_importance_filter:
            # 중요도 필터링 없이 비율만으로 선별
            # 중요도 점수는 정렬에만 사용 (참고용)
            pass
        else:
            # 기존 방식: 최소 중요도 이상만 선택
            if min_importance is not None:
                filtered_df = df[df['importance_score'] >= min_importance].copy()
                
                if len(filtered_df) == 0:
                    # 중요도 기준을 만족하는 발언이 없으면 점수 상위 N개 선택
                    df = df.sort_values('importance_score', ascending=False)
                    return df.head(max_speeches)
                
                df = filtered_df
        
        if party_balance:
            # 정당별 균형 유지하면서 선택
            selected_indices = set()
            parties = df['party'].dropna().unique()
            if len(parties) > 0:
                speeches_per_party = max(1, max_speeches // len(parties))
                
                for party in parties:
                    party_speeches = df[df['party'] == party].copy()
                    # 중요도 점수는 참고용으로만 사용 (정렬)
                    if not skip_importance_filter:
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
                if not skip_importance_filter:
                    all_available = all_available.sort_values(
                        'importance_score', ascending=False
                    )
                additional_indices = all_available.head(remaining).index.tolist()
                selected_indices.update(additional_indices)
            
            result = df[df.index.isin(selected_indices)].copy()
            if not skip_importance_filter:
                result = result.sort_values('importance_score', ascending=False)
            return result.head(max_speeches)
        else:
            # 단순 선택 (중요도 점수는 참고용)
            if not skip_importance_filter:
                df = df.sort_values('importance_score', ascending=False)
            return df.head(max_speeches)

    def prepare_session_summary_payload(
        self, 
        df: pd.DataFrame,
        *,
        max_sample_speeches: Optional[int] = None,
        sample_ratio: float = 0.8,
        min_samples: int = 50,
        max_samples: Optional[int] = None,
        max_chars_per_speech: int = 500,
        skip_importance_filter: bool = True,
        qa_pairs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """회차 요약을 위한 데이터 준비 (대부분의 데이터 포함).
        
        Parameters
        ----------
        df : pd.DataFrame
            세션 발언 데이터 (품질 필터링 후)
        max_sample_speeches : Optional[int], optional
            최대 샘플 발언 개수 (고정값). None이면 sample_ratio 사용
        sample_ratio : float, default 0.8
            전체 발언 대비 샘플 비율 (0.0 ~ 1.0). 기본값 80%로 설정하여 
            대부분의 데이터를 분석에 포함
        min_samples : int, default 50
            최소 샘플 발언 개수 (비율 기반 계산 시)
        max_samples : Optional[int], optional
            최대 샘플 발언 개수 (비율 기반 계산 시). None이면 제한 없음
        max_chars_per_speech : int, default 500
            각 발언 샘플의 최대 문자 수
        skip_importance_filter : bool, default True
            True면 중요도 필터링을 건너뛰고 비율만으로 선별
        qa_pairs : Optional[List[Dict[str, str]]], optional
            질의-응답 쌍 리스트
        """
        # 통계 계산
        agenda_stats: Dict[str, int] = {}
        for agenda in df["agenda_item_titles"].dropna().unique():
            if pd.notna(agenda) and str(agenda).strip():
                agenda_count = len(df[df["agenda_item_titles"] == agenda])
                agenda_stats[str(agenda)] = agenda_count

        party_stats = df["party"].value_counts().to_dict()
        
        # 발언 길이 통계 계산
        speech_lengths = df["speech_text"].astype(str).str.len()
        avg_speech_length = float(speech_lengths.mean()) if len(speech_lengths) > 0 else 0.0
        
        # QA 쌍 수 계산
        qa_pairs_count = len(qa_pairs) if qa_pairs else 0
        
        # 발언 선별 (중요도 필터링 없이 비율만으로)
        important_speeches = self.select_important_speeches(
            df,
            max_speeches=max_sample_speeches,
            sample_ratio=sample_ratio,
            min_samples=min_samples,
            max_samples=max_samples,
            skip_importance_filter=skip_importance_filter,
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
            "qa_pairs_count": qa_pairs_count,
            "avg_speech_length": avg_speech_length,
        }

    def prepare_session_summary_payloads_batched(
        self, 
        df: pd.DataFrame,
        *,
        sample_ratio: float = 0.8,
        batch_size: int = 1000,
        max_chars_per_speech: int = 500,
        skip_importance_filter: bool = True,
        qa_pairs: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """회차 요약을 위한 데이터를 배치로 준비 (토큰 제한 회피용).
        
        Parameters
        ----------
        df : pd.DataFrame
            세션 발언 데이터 (품질 필터링 후)
        sample_ratio : float, default 0.8
            전체 발언 대비 샘플 비율 (0.0 ~ 1.0)
        batch_size : int, default 1000
            각 배치당 최대 발언 개수 (토큰 제한 고려)
        max_chars_per_speech : int, default 500
            각 발언 샘플의 최대 문자 수
        skip_importance_filter : bool, default True
            True면 중요도 필터링을 건너뛰고 비율만으로 선별
        qa_pairs : Optional[List[Dict[str, str]]], optional
            질의-응답 쌍 리스트
        
        Returns
        -------
        List[Dict[str, Any]]
            배치별 payload 리스트
        """
        # 전체 통계 계산 (한 번만)
        agenda_stats: Dict[str, int] = {}
        for agenda in df["agenda_item_titles"].dropna().unique():
            if pd.notna(agenda) and str(agenda).strip():
                agenda_count = len(df[df["agenda_item_titles"] == agenda])
                agenda_stats[str(agenda)] = agenda_count

        party_stats = df["party"].value_counts().to_dict()
        speech_lengths = df["speech_text"].astype(str).str.len()
        avg_speech_length = float(speech_lengths.mean()) if len(speech_lengths) > 0 else 0.0
        qa_pairs_count = len(qa_pairs) if qa_pairs else 0
        
        # 선별할 발언 개수 계산
        total_to_select = int(len(df) * sample_ratio)
        
        # 발언 선별 (중요도 필터링 없이)
        selected_speeches = self.select_important_speeches(
            df,
            max_speeches=total_to_select,
            skip_importance_filter=skip_importance_filter,
            party_balance=True
        )
        
        # 배치로 나누기
        batches: List[Dict[str, Any]] = []
        total_selected = len(selected_speeches)
        
        for i in range(0, total_selected, batch_size):
            batch_speeches = selected_speeches.iloc[i:i + batch_size]
            
            # 배치별 샘플 데이터 구성
            speeches_sample: List[Dict[str, Any]] = []
            for _, row in batch_speeches.iterrows():
                speeches_sample.append({
                    "party": row.get("party", ""),
                    "speaker": row.get("speaker_name", ""),
                    "text": str(row.get("speech_text", ""))[:max_chars_per_speech],
                })
            
            batches.append({
                "total_speeches": len(df),
                "quality_speeches": total_selected,  # 전체 선별 개수
                "batch_speeches": len(batch_speeches),  # 이 배치의 발언 수
                "batch_index": i // batch_size,  # 배치 인덱스 (0부터 시작)
                "total_batches": (total_selected + batch_size - 1) // batch_size,  # 전체 배치 수
                "agenda_stats": agenda_stats,
                "party_stats": party_stats,
                "speeches_sample": speeches_sample,
                "qa_pairs_count": qa_pairs_count,
                "avg_speech_length": avg_speech_length,
            })
        
        return batches

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
        self, 
        session_name: str, 
        *, 
        payload: Dict[str, Any],
        db_client: Optional[Any] = None,
    ) -> Optional[SessionSummary]:
        """세션 요약 실행 (이전 회차 정보 조회 포함)."""
        previous_session_summary = None
        
        # 이전 회차 정보 조회
        if db_client:
            try:
                # 회차 번호 추출: "제415회" -> 415
                import re
                match = re.search(r'제(\d+)회', session_name)
                if match:
                    current_number = int(match.group(1))
                    previous_number = current_number - 1
                    previous_session_name = f"제{previous_number}회"
                    
                    # DB에서 이전 회차 조회
                    previous_session_record = db_client.get_session_record(previous_session_name)
                    if previous_session_record:
                        # 이전 회차 정보 추출
                        metadata = previous_session_record.get("metadata", {})
                        session_summary_data = metadata.get("session_summary", {})
                        session_summary_metadata = session_summary_data.get("metadata", {})
                        
                        previous_session_summary = {
                            "session_name": previous_session_name,
                            "summary_text": previous_session_record.get("summary_text", ""),
                            "key_issues": session_summary_data.get("key_issues", []),
                            "party_positions_overview": session_summary_metadata.get("party_positions_overview", {}),
                            "quantitative_stats": session_summary_metadata.get("quantitative_stats"),
                        }
            except Exception as e:
                # 이전 회차 조회 실패 시 무시 (첫 회차일 수 있음)
                pass
        
        return self.analyzer.analyze_session_summary(
            session_name=session_name,
            session_payload=payload,
            previous_session_summary=previous_session_summary,
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

    @staticmethod
    def merge_batch_summaries(
        batch_summaries: List[SessionSummary],
        session_name: str
    ) -> Optional[SessionSummary]:
        """여러 배치의 SessionSummary를 하나로 통합.
        
        Parameters
        ----------
        batch_summaries : List[SessionSummary]
            통합할 배치별 요약 리스트
        session_name : str
            세션 이름
        
        Returns
        -------
        Optional[SessionSummary]
            통합된 요약 (배치가 없으면 None)
        """
        if not batch_summaries:
            return None
        
        if len(batch_summaries) == 1:
            return batch_summaries[0]
        
        # 1. key_issues 통합 (중복 제거 및 중요도 기준 정렬)
        all_key_issues: Dict[str, Dict[str, Any]] = {}
        for summary in batch_summaries:
            if summary.key_issues:
                for issue in summary.key_issues:
                    issue_text = issue.get("issue", "")
                    if issue_text:
                        # 이미 존재하면 중요도가 높은 것으로 업데이트
                        if issue_text not in all_key_issues:
                            all_key_issues[issue_text] = issue.copy()
                        else:
                            existing_importance = all_key_issues[issue_text].get("importance", "낮음")
                            new_importance = issue.get("importance", "낮음")
                            importance_order = {"높음": 3, "중간": 2, "낮음": 1}
                            if importance_order.get(new_importance, 1) > importance_order.get(existing_importance, 1):
                                all_key_issues[issue_text] = issue.copy()
        
        # 중요도 순으로 정렬
        importance_order = {"높음": 3, "중간": 2, "낮음": 1}
        merged_key_issues = sorted(
            all_key_issues.values(),
            key=lambda x: importance_order.get(x.get("importance", "낮음"), 1),
            reverse=True
        )
        
        # 2. overall_sentiment 평균 계산
        sentiments = [s.overall_sentiment for s in batch_summaries if s.overall_sentiment is not None]
        merged_sentiment = sum(sentiments) / len(sentiments) if sentiments else None
        
        # 3. raw_summary 통합 (각 배치 요약을 연결)
        raw_summaries = [s.raw_summary for s in batch_summaries if s.raw_summary]
        merged_raw_summary = "\n\n".join(raw_summaries) if raw_summaries else None
        
        # 4. metadata 통합
        merged_metadata: Dict[str, Any] = {}
        
        # party_positions_overview 통합
        all_party_positions: Dict[str, Dict[str, Any]] = {}
        for summary in batch_summaries:
            party_positions = summary.metadata.get("party_positions_overview", {})
            if isinstance(party_positions, dict):
                for party, position in party_positions.items():
                    if party not in all_party_positions:
                        all_party_positions[party] = position.copy()
                    else:
                        # 기존 정보와 병합 (리스트는 합치기)
                        existing = all_party_positions[party]
                        if isinstance(position, dict) and isinstance(existing, dict):
                            for key, value in position.items():
                                if key in existing:
                                    if isinstance(existing[key], list) and isinstance(value, list):
                                        existing[key].extend(value)
                                    elif isinstance(existing[key], str) and isinstance(value, str):
                                        existing[key] = f"{existing[key]}; {value}"
                                else:
                                    existing[key] = value
        merged_metadata["party_positions_overview"] = all_party_positions
        
        # quantitative_stats 통합 (평균 또는 합계)
        all_stats: Dict[str, Any] = {}
        for summary in batch_summaries:
            stats = summary.metadata.get("quantitative_stats", {})
            if isinstance(stats, dict):
                for key, value in stats.items():
                    if key not in all_stats:
                        all_stats[key] = value
                    else:
                        # 숫자면 합계, 리스트면 합치기
                        if isinstance(value, (int, float)) and isinstance(all_stats[key], (int, float)):
                            all_stats[key] = all_stats[key] + value
                        elif isinstance(value, list) and isinstance(all_stats[key], list):
                            all_stats[key].extend(value)
        merged_metadata["quantitative_stats"] = all_stats
        
        # major_conflicts 통합
        all_conflicts: List[str] = []
        for summary in batch_summaries:
            conflicts = summary.metadata.get("major_conflicts", [])
            if isinstance(conflicts, list):
                all_conflicts.extend(conflicts)
        merged_metadata["major_conflicts"] = list(set(all_conflicts))  # 중복 제거
        
        # key_events 통합
        all_events: List[str] = []
        for summary in batch_summaries:
            events = summary.metadata.get("key_events", [])
            if isinstance(events, list):
                all_events.extend(events)
        merged_metadata["key_events"] = list(set(all_events))  # 중복 제거
        
        # trend_analysis 통합 (마지막 배치의 트렌드 사용, 또는 모두 통합)
        trend_analyses = [s.metadata.get("trend_analysis") for s in batch_summaries if s.metadata.get("trend_analysis")]
        if trend_analyses:
            merged_metadata["trend_analysis"] = trend_analyses[-1]  # 마지막 배치 사용
        
        # quantitative_insights 통합
        all_insights: List[str] = []
        for summary in batch_summaries:
            insights = summary.metadata.get("quantitative_insights", [])
            if isinstance(insights, list):
                all_insights.extend(insights)
        merged_metadata["quantitative_insights"] = list(set(all_insights))  # 중복 제거
        
        # raw_llm_response는 마지막 배치 사용
        last_llm_response = batch_summaries[-1].metadata.get("raw_llm_response")
        if last_llm_response:
            merged_metadata["raw_llm_response"] = last_llm_response
        
        # 배치 정보 추가
        merged_metadata["batch_info"] = {
            "total_batches": len(batch_summaries),
            "merged_at": pd.Timestamp.now().isoformat()
        }
        
        # 통합된 SessionSummary 생성
        merged_summary = SessionSummary(
            session_name=session_name,
            meeting_date=batch_summaries[0].meeting_date,  # 첫 번째 배치의 날짜 사용
            key_issues=merged_key_issues,
            overall_sentiment=merged_sentiment,
            raw_summary=merged_raw_summary,
            metadata=merged_metadata
        )
        
        return merged_summary


