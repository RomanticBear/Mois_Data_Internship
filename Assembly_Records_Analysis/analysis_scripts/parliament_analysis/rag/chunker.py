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


