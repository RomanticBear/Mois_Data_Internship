"""
[역할] 분석 결과를 Supabase에 저장하고 벡터화
- persist_analysis_to_supabase(): 메인 저장 함수
  - 세션 요약 임베딩 생성 및 저장
  - 안건별 임베딩 생성 및 저장
  - 정당별 입장 임베딩 생성 및 저장
  - QA 질문/답변 임베딩 생성 및 저장
  - RAG 문서 청킹 및 벡터 저장
- EmbeddingClient를 사용하여 텍스트를 벡터로 변환
- VectorStore를 사용하여 벡터를 Supabase에 저장
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Sequence
from uuid import uuid4

import pandas as pd

from ..analysis.models import AgendaPartyAnalysis, PartyPosition, QAAnalysisMetrics, SessionSummary
from ..data.db_client import SupabaseDBClient
from ..data.embedding_client import EmbeddingClient
from ..rag.chunker import RAGChunker
from ..rag.vector_store import VectorItem, VectorStore


def party_position_to_row(
    position: PartyPosition,
    agenda_id: str,
    embedding_client: EmbeddingClient,
) -> Dict[str, Any]:
    summary_components: List[str] = []
    if position.summary_text:
        summary_components.append(position.summary_text)
    if position.key_points:
        summary_components.extend(position.key_points)
    if position.concerns:
        summary_components.extend(position.concerns)
    if position.suggestions:
        summary_components.extend(position.suggestions)
    summary_text = position.summary_text or "\n".join(summary_components)
    stance_embedding = (
        embedding_client.embed_text(summary_text) if summary_text else None
    )

    row: Dict[str, Any] = {
        "agenda_id": agenda_id,
        "party_name": position.party_name,
        "stance_label": position.stance_label,
        "key_points": position.key_points,
        "concerns": position.concerns,
        "suggestions": position.suggestions,
        "metadata": position.metadata,
    }
    if summary_text:
        row["summary_text"] = summary_text
    if stance_embedding:
        row["stance_embedding"] = stance_embedding
    return row


def build_party_position_payload(
    analysis: AgendaPartyAnalysis,
    agenda_id: str,
    embedding_client: EmbeddingClient,
) -> Iterable[Dict[str, Any]]:
    for position in analysis.party_positions:
        yield party_position_to_row(position, agenda_id, embedding_client)


def _documents_to_vector_items(documents: Iterable[Dict[str, Any]]) -> List[VectorItem]:
    items: List[VectorItem] = []
    for doc in documents:
        metadata = dict(doc.get("metadata", {}))
        metadata["source_id"] = doc.get("source_id")
        metadata["chunk_index"] = doc.get("chunk_index")
        items.append(
            VectorItem(
                record_id=f"{doc.get('source_id')}:{doc.get('chunk_index')}",
                content=doc.get("content", ""),
                metadata=metadata,
            )
        )
    return items


def persist_analysis_to_supabase(
    *,
    session_name: str,
    hash_digest: str,
    analysis_version: str,
    raw_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    session_summary: SessionSummary | None,
    party_analyses: Sequence[AgendaPartyAnalysis],
    qa_pairs: Sequence[Dict[str, str]],
    qa_metrics: QAAnalysisMetrics | None,
    db_client: SupabaseDBClient,
    embedding_client: EmbeddingClient,
    vector_store: VectorStore,
    chunker: RAGChunker,
) -> None:
    summary_text = ""
    if session_summary:
        summary_text = session_summary.raw_summary or ""
        if not summary_text and session_summary.key_issues:
            summary_text = "\n".join(
                f"{issue.get('issue')} - {issue.get('description')}"
                for issue in session_summary.key_issues
            )
    summary_embedding = (
        embedding_client.embed_text(summary_text) if summary_text else None
    )

    session_metadata: Dict[str, Any] = {
        "total_speeches": len(raw_df),
        "quality_speeches": len(quality_df),
        "analysis_version": analysis_version,
    }
    if session_summary:
        session_metadata["session_summary"] = asdict_no_embedding(session_summary)
    if qa_metrics:
        session_metadata["qa_metrics"] = asdict_no_embedding(qa_metrics)

    sanitized_metadata = json.loads(
        json.dumps(session_metadata, ensure_ascii=False, default=str)
    )

    session_record: Dict[str, Any] = {
        "session_name": session_name,
        "hash_digest": hash_digest,
        "analysis_version": analysis_version,
        "metadata": sanitized_metadata,
    }
    if summary_text:
        session_record["summary_text"] = summary_text
    if summary_embedding:
        session_record["summary_embedding"] = summary_embedding
    if session_summary and session_summary.metadata.get("meeting_date"):
        session_record["meeting_date"] = session_summary.metadata["meeting_date"]

    session_row = db_client.upsert_session_record(session_record)
    session_id = session_row["session_id"]

    # Cleanup existing artifacts for idempotent reruns
    existing_agenda_ids = db_client.get_agenda_ids_for_session(session_id)
    if existing_agenda_ids:
        db_client.delete_party_positions_for_agendas(existing_agenda_ids)
        db_client.delete_qa_for_agendas(existing_agenda_ids)
    db_client.delete_agenda_items(session_id)
    db_client.delete_issue_trends(session_id)
    db_client.delete_rag_documents_by_session(session_name=session_name)

    # Agenda items
    agenda_rows: List[Dict[str, Any]] = []
    for analysis in party_analyses:
        agenda_summary = analysis.summary_text or ""
        if not agenda_summary:
            bullet_points = analysis.consensus_points + analysis.conflict_points
            agenda_summary = "\n".join(bullet_points)
        agenda_embedding = (
            embedding_client.embed_text(agenda_summary) if agenda_summary else None
        )
        agenda_payload: Dict[str, Any] = {
            "session_id": session_id,
            "agenda_title": analysis.agenda_title,
            "summary_text": agenda_summary,
            "metadata": {
                "consensus_points": analysis.consensus_points,
                "conflict_points": analysis.conflict_points,
                "cooperation_level": analysis.cooperation_level,
                "summary_text": analysis.summary_text,
                "metadata": analysis.metadata,
            },
        }
        if agenda_embedding:
            agenda_payload["summary_embedding"] = agenda_embedding
        agenda_rows.append(agenda_payload)

    agenda_response = db_client.upsert_agenda_items(agenda_rows)
    agenda_lookup = {
        row["agenda_title"]: row["agenda_id"] for row in agenda_response if "agenda_id" in row
    }

    # Party positions
    party_position_rows: List[Dict[str, Any]] = []
    for analysis in party_analyses:
        agenda_id = agenda_lookup.get(analysis.agenda_title)
        if not agenda_id:
            continue
        party_position_rows.extend(
            build_party_position_payload(analysis, agenda_id, embedding_client)
        )
    db_client.upsert_party_positions(party_position_rows)

    # QA interactions
    if qa_pairs:
        question_embeddings = embedding_client.embed_texts(
            [pair.get("question", "") for pair in qa_pairs]
        )
        answer_embeddings = embedding_client.embed_texts(
            [pair.get("answer", "") for pair in qa_pairs]
        )
        qa_rows: List[Dict[str, Any]] = []
        for pair, q_embed, a_embed in zip(qa_pairs, question_embeddings, answer_embeddings):
            qa_rows.append(
                {
                    "qa_id": str(uuid4()),
                    "agenda_id": agenda_lookup.get(pair.get("agenda_title", ""), None),
                    "questioner": pair.get("questioner"),
                    "respondent": pair.get("answerer"),
                    "question_text": pair.get("question"),
                    "answer_text": pair.get("answer"),
                    "effectiveness_score": None,
                    "effectiveness_bucket": None,
                    "tags": [],
                    "metadata": {
                        "question_party": pair.get("question_party"),
                        "answer_party": pair.get("answer_party"),
                        "session_name": session_name,
                    },
                    "question_embedding": q_embed,
                    "answer_embedding": a_embed,
                }
            )
        db_client.upsert_qa_interactions(qa_rows)

    # RAG documents
    vector_items: List[VectorItem] = []

    if session_summary:
        summary_docs = chunker.chunk_session_summary(
            asdict_no_embedding(session_summary),
            session_name=session_name,
        )
        vector_items.extend(_documents_to_vector_items(summary_docs))

    party_docs = chunker.chunk_party_positions(
        [
            asdict_no_embedding(position)
            for analysis in party_analyses
            for position in analysis.party_positions
        ],
        agenda_id_lookup=agenda_lookup,
        session_name=session_name,
    )
    vector_items.extend(_documents_to_vector_items(party_docs))

    qa_docs = list(
        chunker.chunk_qa_pairs(
            qa_pairs,
            agenda_id_lookup=agenda_lookup,
            session_name=session_name,
        )
    )
    vector_items.extend(_documents_to_vector_items(qa_docs))
    vector_store.upsert_documents(vector_items)


def asdict_no_embedding(obj: Any) -> Dict[str, Any]:
    """Safely convert dataclasses to dict without custom serialization."""
    if hasattr(obj, "__dataclass_fields__"):
        return json.loads(json.dumps(asdict(obj), ensure_ascii=False, default=str))
    return obj


