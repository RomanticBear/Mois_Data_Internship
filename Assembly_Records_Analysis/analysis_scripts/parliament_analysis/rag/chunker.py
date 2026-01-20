"""
[역할] RAG용 문서 청킹
- chunk_session_summary(): 세션 요약 청킹
- chunk_party_positions(): 정당 입장 청킹
- chunk_qa_pairs(): QA 페어 청킹
- _split_text(): 텍스트를 청크 크기로 분할 (overlap 포함)
- RAG 검색을 위해 긴 문서를 작은 청크로 분할하여 메타데이터와 함께 저장
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Sequence


class RAGChunker:
    """Generate consistent chunk payloads from transcripts and summaries."""

    def __init__(self, *, chunk_size: int = 800, overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Session-level chunks
    # ------------------------------------------------------------------

    def chunk_session_summary(
        self, session_summary: dict, *, session_name: str
    ) -> List[Dict[str, object]]:
        """Create RAG documents for session-level summaries."""
        session_name = session_summary.get("session_name")
        key_issues = session_summary.get("key_issues", [])
        major_conflicts = session_summary.get("metadata", {}).get("major_conflicts", [])
        key_events = session_summary.get("metadata", {}).get("key_events", [])

        summary_parts: List[str] = []
        if session_summary.get("raw_summary"):
            summary_parts.append(str(session_summary["raw_summary"]))

        if key_issues:
            summary_parts.append("핵심 이슈 요약:")
            for issue in key_issues:
                summary_parts.append(
                    f"- {issue.get('issue')} ({issue.get('importance')}) : {issue.get('description')}"
                )

        if major_conflicts:
            summary_parts.append("주요 쟁점:")
            for conflict in major_conflicts:
                summary_parts.append(
                    f"- {conflict.get('topic')} / 참여 정당: {', '.join(conflict.get('parties_involved', []))}"
                )

        if key_events:
            summary_parts.append("주요 사건:")
            for event in key_events:
                summary_parts.append(
                    f"- {event.get('event')} : {event.get('description')} / 대응 {event.get('response')}"
                )

        # 정당별 관심사 추가 (확장)
        party_positions_overview = session_summary.get("metadata", {}).get("party_positions_overview", {})
        if party_positions_overview:
            summary_parts.append("\n정당별 주요 관심사:")
            for party_name, party_data in party_positions_overview.items():
                if isinstance(party_data, dict):
                    main_concerns = party_data.get("main_concerns", [])
                    stance = party_data.get("stance", "")
                    key_statements = party_data.get("key_statements", "")
                    party_text = f"\n[{party_name}] 입장: {stance}"
                    if main_concerns:
                        party_text += f"\n주요 관심사: {', '.join(main_concerns)}"
                    if key_statements:
                        party_text += f"\n주요 발언: {key_statements}"
                    summary_parts.append(party_text)

        # 트렌드 분석 섹션 추가 (있는 경우만)
        trend_analysis = session_summary.get("metadata", {}).get("trend_analysis")
        if trend_analysis and isinstance(trend_analysis, dict):
            summary_parts.append("\n=== 이전 회차 대비 변화 추이 ===")
            issue_changes = trend_analysis.get("issue_changes", [])
            if issue_changes:
                summary_parts.append("\n이슈별 변화:")
                for change in issue_changes:
                    issue = change.get("issue", "")
                    change_type = change.get("change", "")
                    prev_count = change.get("previous_mention_count", 0)
                    curr_count = change.get("current_mention_count", 0)
                    description = change.get("description", "")
                    summary_parts.append(f"  - {issue}: {change_type} ({prev_count}회 → {curr_count}회) - {description}")
            party_position_changes = trend_analysis.get("party_position_changes", {})
            if party_position_changes:
                summary_parts.append("\n정당별 입장 변화:")
                for party, change_data in party_position_changes.items():
                    if isinstance(change_data, dict):
                        issue = change_data.get("issue", "")
                        change_type = change_data.get("change", "")
                        description = change_data.get("description", "")
                        summary_parts.append(f"  - {party} ({issue}): {change_type} - {description}")
            quantitative_changes = trend_analysis.get("quantitative_changes", {})
            if quantitative_changes:
                summary_parts.append("\n정량적 변화:")
                speech_change = quantitative_changes.get("speech_count_change", "")
                question_change = quantitative_changes.get("question_count_change", "")
                description = quantitative_changes.get("description", "")
                if speech_change:
                    summary_parts.append(f"  - 발언 수: {speech_change}")
                if question_change:
                    summary_parts.append(f"  - 질문 수: {question_change}")
                if description:
                    summary_parts.append(f"  - 설명: {description}")

        # 정량적 통계 섹션 추가
        quantitative_stats = session_summary.get("metadata", {}).get("quantitative_stats", {})
        if quantitative_stats:
            summary_parts.append("\n=== 정량적 통계 ===")
            total_speeches = quantitative_stats.get("total_speeches", 0)
            qa_pairs_count = quantitative_stats.get("qa_pairs_count", 0)
            avg_speech_length = quantitative_stats.get("avg_speech_length", 0.0)
            summary_parts.append(f"총 발언 수: {total_speeches:,}개")
            summary_parts.append(f"QA 쌍 수: {qa_pairs_count}개")
            summary_parts.append(f"평균 발언 길이: {avg_speech_length:.0f}자")
            issue_mentions = quantitative_stats.get("issue_mentions", {})
            if issue_mentions:
                summary_parts.append("\n이슈별 언급 횟수:")
                sorted_issues = sorted(issue_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
                for issue, count in sorted_issues:
                    summary_parts.append(f"  - {issue}: {count}회")

        # 정량적 인사이트 섹션 추가
        quantitative_insights = session_summary.get("metadata", {}).get("quantitative_insights", {})
        if quantitative_insights:
            summary_parts.append("\n=== 정량적 인사이트 ===")
            issue_ranking = quantitative_insights.get("issue_importance_ranking", [])
            if issue_ranking:
                summary_parts.append(f"주요 이슈 순위: {', '.join(issue_ranking)}")
            most_active_party = quantitative_insights.get("most_active_party", "")
            if most_active_party:
                summary_parts.append(f"가장 활발한 정당: {most_active_party}")
            key_statistics = quantitative_insights.get("key_statistics", "")
            if key_statistics:
                summary_parts.append(f"주요 통계 해석: {key_statistics}")

        text = "\n".join(part for part in summary_parts if part)
        metadata = {
            "source_type": "session_summary",
            "session_name": session_name,
        }
        return self._build_chunk_payloads(
            text=text,
            source_id=f"session::{session_name}",
            metadata=metadata,
        )

    def chunk_party_positions(
        self,
        positions: Sequence[dict],
        *,
        agenda_id_lookup: Dict[str, str],
        session_name: str,
    ) -> List[Dict[str, object]]:
        """Create RAG documents from party stance bullet points."""
        documents: List[Dict[str, object]] = []
        for position in positions:
            agenda_title = position.get("agenda_title")
            agenda_id = agenda_id_lookup.get(agenda_title or "", "")
            summary_text = position.get("summary_text") or ""

            bullet_sections = []
            key_points = position.get("key_points") or []
            if key_points:
                bullet_sections.append("주요 포인트:\n" + "\n".join(f"- {p}" for p in key_points))

            concerns = position.get("concerns") or []
            if concerns:
                bullet_sections.append("우려 사항:\n" + "\n".join(f"- {c}" for c in concerns))

            suggestions = position.get("suggestions") or []
            if suggestions:
                bullet_sections.append("제안 사항:\n" + "\n".join(f"- {s}" for s in suggestions))

            stance_label = position.get("stance_label")
            party_name = position.get("party_name")

            text_sections = [
                f"[안건] {agenda_title}",
                f"[정당] {party_name}",
                f"[입장] {stance_label}",
                summary_text,
                "\n\n".join(bullet_sections),
            ]
            text = "\n".join(filter(None, text_sections))

            chunk_metadata = {
                "source_type": "party_position",
                "agenda_title": agenda_title,
                "agenda_id": agenda_id,
                "session_name": session_name,
                "party_name": party_name,
                "stance_label": stance_label,
            }
            documents.extend(
                self._build_chunk_payloads(
                    text=text,
                    source_id=f"session::{session_name}::agenda::{agenda_title}::party::{party_name}",
                    metadata=chunk_metadata,
                )
            )
        return documents

    def chunk_qa_pairs(
        self,
        qa_pairs: Iterable[dict],
        *,
        agenda_id_lookup: Dict[str, str],
        session_name: str,
    ) -> Iterator[Dict[str, object]]:
        """Yield RAG documents from question-answer pairs."""
        for index, pair in enumerate(qa_pairs):
            agenda_title = pair.get("agenda_title")
            agenda_id = agenda_id_lookup.get(agenda_title or "", "")
            question = pair.get("question", "")
            answer = pair.get("answer", "")

            text = (
                f"[질문자] {pair.get('questioner')} ({pair.get('question_party')})\n"
                f"[질문]\n{question}\n\n"
                f"[답변자] {pair.get('answerer')} ({pair.get('answer_party')})\n"
                f"[답변]\n{answer}"
            )

            metadata = {
                "source_type": "qa_pair",
                "agenda_title": agenda_title,
                "agenda_id": agenda_id,
                "session_name": session_name,
                "questioner": pair.get("questioner"),
                "respondent": pair.get("answerer"),
                "effectiveness_bucket": pair.get("effectiveness_bucket"),
            }

            for chunk in self._build_chunk_payloads(
                text=text,
                source_id=f"session::{session_name}::qa::{index}",
                metadata=metadata,
            ):
                yield chunk

    def chunk_agenda_analysis(
        self,
        agenda_analysis: dict,
        *,
        agenda_id_lookup: Dict[str, str],
        session_name: str,
    ) -> List[Dict[str, object]]:
        """Create RAG documents for agenda-level analysis (합의점/대립점)."""
        agenda_title = agenda_analysis.get("agenda_title", "")
        agenda_id = agenda_id_lookup.get(agenda_title, "")
        consensus_points = agenda_analysis.get("consensus_points", [])
        conflict_points = agenda_analysis.get("conflict_points", [])
        cooperation_level = agenda_analysis.get("cooperation_level", "")

        text_sections = [
            f"[안건] {agenda_title}",
            f"[협력 수준] {cooperation_level}",
        ]

        if consensus_points:
            text_sections.append("\n[합의점]")
            text_sections.extend(f"- {point}" for point in consensus_points)

        if conflict_points:
            text_sections.append("\n[대립점]")
            text_sections.extend(f"- {point}" for point in conflict_points)

        text = "\n".join(text_sections)

        metadata = {
            "source_type": "agenda_analysis",
            "agenda_title": agenda_title,
            "agenda_id": agenda_id,
            "session_name": session_name,
            "cooperation_level": cooperation_level,
        }

        return self._build_chunk_payloads(
            text=text,
            source_id=f"session::{session_name}::agenda::{agenda_title}::analysis",
            metadata=metadata,
        )

    def chunk_qa_metrics(
        self,
        qa_metrics: dict,
        *,
        session_name: str,
    ) -> List[Dict[str, object]]:
        """Create RAG documents for QA quality metrics."""
        quality_distribution = qa_metrics.get("quality_distribution", {})
        question_types = qa_metrics.get("question_types", {})
        answer_quality = qa_metrics.get("answer_quality", {})
        key_issues = qa_metrics.get("key_issues", [])

        text_sections = [
            f"[세션] {session_name}",
            "\n[QA 품질 통계]",
        ]

        if quality_distribution:
            text_sections.append("품질 분포:")
            text_sections.append(f"- 고품질: {quality_distribution.get('high', 0)}%")
            text_sections.append(f"- 중품질: {quality_distribution.get('medium', 0)}%")
            text_sections.append(f"- 저품질: {quality_distribution.get('low', 0)}%")

        if question_types:
            text_sections.append("\n질문 유형 분포:")
            for q_type, percentage in question_types.items():
                type_name = {
                    "policy_inquiry": "정책 질의",
                    "fact_checking": "사실 확인",
                    "criticism": "비판 질의",
                    "suggestion": "제안 질의",
                }.get(q_type, q_type)
                text_sections.append(f"- {type_name}: {percentage}%")

        if answer_quality:
            text_sections.append("\n답변 품질 지표:")
            text_sections.append(f"- 완성도: {answer_quality.get('completeness', 'N/A')}/10")
            text_sections.append(f"- 구체성: {answer_quality.get('specificity', 'N/A')}/10")
            text_sections.append(f"- 응답성: {answer_quality.get('responsiveness', 'N/A')}/10")

        if key_issues:
            text_sections.append("\n주요 이슈별 QA:")
            for issue in key_issues:
                text_sections.append(
                    f"- {issue.get('issue')}: {issue.get('qa_count', 0)}건 "
                    f"(평균 품질: {issue.get('quality', 'N/A')})"
                )

        text = "\n".join(text_sections)

        metadata = {
            "source_type": "qa_metrics",
            "session_name": session_name,
        }

        return self._build_chunk_payloads(
            text=text,
            source_id=f"session::{session_name}::qa_metrics",
            metadata=metadata,
        )

    def chunk_original_speeches(
        self,
        speeches_df: "pd.DataFrame",  # type: ignore
        *,
        session_name: str,
        min_importance: float = 0.3,
        max_speeches: int = 100,
    ) -> List[Dict[str, object]]:
        """원본 발언을 직접 청킹하여 RAG 문서 생성 (하이브리드 검색용)"""
        import pandas as pd
        
        # 중요도 점수 계산 (이미 계산되어 있을 수 있음)
        if 'importance_score' not in speeches_df.columns:
            from ..pipeline.workflow import SessionAnalysisWorkflow
            speeches_df = speeches_df.copy()
            speeches_df['importance_score'] = speeches_df.apply(
                lambda row: SessionAnalysisWorkflow.calculate_importance_score(row),
                axis=1
            )
        
        # 중요도 높은 발언만 선별
        important_speeches = speeches_df[
            speeches_df['importance_score'] >= min_importance
        ].sort_values('importance_score', ascending=False).head(max_speeches)
        
        documents: List[Dict[str, object]] = []
        
        for idx, row in important_speeches.iterrows():
            speech_text = str(row.get('speech_text', ''))
            speaker = row.get('speaker_name', '') or row.get('speaker', '')
            party = row.get('party', '')
            agenda_title = row.get('agenda_item_titles', '') or row.get('agenda_title', '')
            
            # 발언 텍스트 구성
            text = f"""
[발언자] {speaker} ({party})
[안건] {agenda_title}
[발언 내용]
{speech_text}
            """.strip()
            
            metadata = {
                "source_type": "original_speech",
                "session_name": session_name,
                "speaker": speaker,
                "party": party,
                "agenda_title": agenda_title,
                "importance_score": float(row.get('importance_score', 0)),
            }
            
            # 청킹
            chunks = self._split_text(text)
            for chunk_idx, chunk_text in enumerate(chunks):
                documents.append({
                    "source_id": f"session::{session_name}::speech::{idx}",
                    "chunk_index": chunk_idx,
                    "content": chunk_text,
                    "metadata": metadata,
                })
        
        return documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chunk_payloads(
        self,
        *,
        text: str,
        source_id: str,
        metadata: Dict[str, object],
    ) -> List[Dict[str, object]]:
        chunks = self._split_text(text)
        documents: List[Dict[str, object]] = []
        for idx, chunk_text in enumerate(chunks):
            documents.append(
                {
                    "source_id": source_id,
                    "chunk_index": idx,
                    "content": chunk_text,
                    "metadata": metadata,
                }
            )
        return documents

    def _split_text(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = max(end - self.overlap, start + 1)
        return chunks


