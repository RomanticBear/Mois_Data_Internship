"""
[역할] Supabase 테이블 스키마 정의
- SESSIONS_TABLE: 세션 메타데이터
- AGENDA_ITEMS_TABLE: 안건 정보
- PARTY_POSITIONS_TABLE: 정당별 입장
- QA_INTERACTIONS_TABLE: QA 상호작용
- ISSUE_TRENDS_TABLE: 이슈 트렌드
- DOCUMENTS_RAG_TABLE: RAG 문서 저장
- TableSchema, Column dataclass로 스키마 구조화
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    constraints: Tuple[str, ...] = ()
    comment: str | None = None


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: Tuple[Column, ...]
    primary_key: Tuple[str, ...]
    unique_constraints: Tuple[Tuple[str, ...], ...] = ()
    indexes: Tuple[Tuple[str, ...], ...] = ()
    comment: str | None = None


SESSIONS_TABLE = TableSchema(
    name="sessions",
    comment="Top-level metadata for each plenary session analysis run.",
    primary_key=("session_id",),
    unique_constraints=(("session_name",),),
    columns=(
        Column("session_id", "uuid", ("default uuid_generate_v4()",), "Primary key."),
        Column("session_name", "text", (), "Human-readable session identifier."),
        Column("meeting_date", "date", (), "Session meeting date."),
        Column("source_path", "text", (), "Filesystem or bucket path to raw transcript."),
        Column("hash_digest", "text", (), "Checksum used to detect content changes."),
        Column("analysis_version", "text", (), "Semantic version for analysis prompts."),
        Column("analyzed_at", "timestamptz", (), "Timestamp of last successful analysis."),
        Column("summary_text", "text", (), "Optional session-level summary."),
        Column("summary_embedding", "vector(1536)", (), "Embedding for semantic search."),
        Column("metadata", "jsonb", (), "Additional attributes or processing info."),
    ),
)

AGENDA_ITEMS_TABLE = TableSchema(
    name="agenda_items",
    comment="Agenda-level summaries and embeddings.",
    primary_key=("agenda_id",),
    columns=(
        Column("agenda_id", "uuid", ("default uuid_generate_v4()",), "Primary key."),
        Column("session_id", "uuid", (), "References sessions.session_id."),
        Column("agenda_title", "text", (), "Agenda item title."),
        Column("agenda_category", "text", (), "Categorized theme."),
        Column("summary_text", "text", (), "LLM summary of the agenda."),
        Column("summary_embedding", "vector(1536)", (), "Embedding for agenda summary."),
        Column("metadata", "jsonb", (), "Additional agenda information."),
    ),
    indexes=(("session_id",), ("agenda_title",)),
)

PARTY_POSITIONS_TABLE = TableSchema(
    name="party_positions",
    comment="Party stance and talking points per agenda.",
    primary_key=("agenda_id", "party_name"),
    columns=(
        Column("agenda_id", "uuid", (), "References agenda_items.agenda_id."),
        Column("party_name", "text", (), "Political party name."),
        Column("stance_label", "text", (), "Label: support / oppose / neutral."),
        Column("key_points", "jsonb", (), "List of supporting arguments."),
        Column("concerns", "jsonb", (), "List of concerns."),
        Column("suggestions", "jsonb", (), "List of suggestions or action items."),
        Column("summary_text", "text", (), "Narrative summary of party stance."),
        Column("stance_embedding", "vector(1536)", (), "Embedding for semantic search."),
        Column("metadata", "jsonb", (), "Extra attributes like speaker counts."),
    ),
)

QA_INTERACTIONS_TABLE = TableSchema(
    name="qa_interactions",
    comment="Question-answer effectiveness metrics and embeddings.",
    primary_key=("qa_id",),
    columns=(
        Column("qa_id", "uuid", ("default uuid_generate_v4()",), "Primary key."),
        Column("agenda_id", "uuid", (), "References agenda_items.agenda_id."),
        Column("questioner", "text", (), "Name of the questioner."),
        Column("respondent", "text", (), "Name of the respondent."),
        Column("question_text", "text", (), "Question transcript."),
        Column("answer_text", "text", (), "Answer transcript."),
        Column("effectiveness_score", "numeric", (), "Score between 0-1 or 0-100."),
        Column("effectiveness_bucket", "text", (), "Categorical label for quick filtering."),
        Column("tags", "jsonb", (), "List of tags such as topic or style."),
        Column("question_embedding", "vector(1536)", (), "Embedding of the question text."),
        Column("answer_embedding", "vector(1536)", (), "Embedding of the answer text."),
        Column("metadata", "jsonb", (), "Additional computed indicators."),
    ),
    indexes=(("agenda_id",), ("questioner",), ("respondent",)),
)

ISSUE_TRENDS_TABLE = TableSchema(
    name="issue_trends",
    comment="Issue frequency and sentiment trends per session.",
    primary_key=("session_id", "issue_keyword"),
    columns=(
        Column("session_id", "uuid", (), "References sessions.session_id."),
        Column("issue_keyword", "text", (), "Normalized issue/topic keyword."),
        Column("sentiment_score", "numeric", (), "Aggregated sentiment score."),
        Column("frequency", "integer", (), "Mention frequency."),
        Column("notes", "text", (), "Supporting context from transcripts."),
        Column("metadata", "jsonb", (), "Extra attributes such as top speakers."),
    ),
    indexes=(("issue_keyword",),),
)

DOCUMENTS_RAG_TABLE = TableSchema(
    name="documents_rag",
    comment="Canonical chunk store for RAG retrieval across analyses.",
    primary_key=("document_id",),
    columns=(
        Column("document_id", "uuid", ("default uuid_generate_v4()",), "Primary key."),
        Column("source_type", "text", (), "One of session_summary / party_position / qa_pair / transcript."),
        Column("source_id", "text", (), "Composite key back to the originating table."),
        Column("chunk_index", "integer", (), "Ordering within the source document."),
        Column("content", "text", (), "Chunk text."),
        Column("metadata", "jsonb", (), "Rich metadata for filtering."),
        Column("embedding", "vector(1536)", (), "Embedding for semantic retrieval."),
    ),
    indexes=(("source_type", "source_id"),),
)


ALL_TABLES: List[TableSchema] = [
    SESSIONS_TABLE,
    AGENDA_ITEMS_TABLE,
    PARTY_POSITIONS_TABLE,
    QA_INTERACTIONS_TABLE,
    ISSUE_TRENDS_TABLE,
    DOCUMENTS_RAG_TABLE,
]






