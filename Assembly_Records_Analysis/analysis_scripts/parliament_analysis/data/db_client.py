"""
[역할] Supabase 데이터베이스 클라이언트 래퍼
- 세션, 안건, 정당 입장, QA, 벡터 문서 등 CRUD 작업
- upsert_session_record(): 세션 레코드 저장/업데이트
- upsert_agenda_items(): 안건 저장
- upsert_party_positions(): 정당 입장 저장
- upsert_qa_interactions(): QA 저장
- upsert_rag_documents(): RAG 문서 저장
- Supabase REST API를 통한 데이터베이스 작업
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from supabase import Client, create_client  # type: ignore
except ImportError as exc:  # pragma: no cover - supabase is optional at dev-time
    Client = Any  # type: ignore
    create_client = None  # type: ignore
    SUPABASE_IMPORT_ERROR = exc
else:
    SUPABASE_IMPORT_ERROR = None


class SupabaseDBClient:
    """Lightweight wrapper around Supabase REST/PostgREST endpoints."""

    def __init__(
        self,
        *,
        client: Client,
        session_table: str = "sessions",
    ) -> None:
        self.client = client
        self.session_table = session_table

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "SupabaseDBClient":
        """Instantiate a Supabase client using environment variables."""
        if SUPABASE_IMPORT_ERROR is not None:
            raise RuntimeError(
                "supabase-py is required to use SupabaseDBClient. "
                "Install with `pip install supabase`."
            ) from SUPABASE_IMPORT_ERROR

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_API_KEY")
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL 및 SUPABASE_SERVICE_ROLE_KEY 환경 변수를 설정해주세요."
            )
        client = create_client(url, key)
        return cls(client=client)

    # ------------------------------------------------------------------
    # Session versioning helpers
    # ------------------------------------------------------------------

    def get_session_record(self, session_name: str) -> Optional[Dict[str, Any]]:
        """Fetch existing session row."""
        response = (
            self.client.table(self.session_table)
            .select("*")
            .eq("session_name", session_name)
            .limit(1)
            .execute()
        )
        data = getattr(response, "data", None) or []
        return data[0] if data else None

    def upsert_session_record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a session entry."""
        payload = payload.copy()
        payload.setdefault("analyzed_at", datetime.now(timezone.utc).isoformat())
        response = (
            self.client.table(self.session_table)
            .upsert(payload, on_conflict="session_name", returning="representation")
            .execute()
        )
        data = getattr(response, "data", None)
        if not data:
            raise RuntimeError("Failed to upsert session record.")
        return data[0]

    def get_sessions_to_process(
        self,
        *,
        session_metadata: Iterable[Dict[str, Any]],
        analysis_version: str,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Compare incoming sessions against stored hash/version to filter worklist."""
        to_process: List[Dict[str, Any]] = []
        for payload in session_metadata:
            session_name = payload["session_name"]
            existing = self.get_session_record(session_name)
            if force or existing is None:
                to_process.append(payload)
                continue

            new_hash = payload.get("hash_digest")
            if not new_hash:
                to_process.append(payload)
                continue

            if (
                existing.get("hash_digest") != new_hash
                or existing.get("analysis_version") != analysis_version
            ):
                to_process.append(payload)
        return to_process

    # ------------------------------------------------------------------
    # Batch upserts for analysis artifacts
    # ------------------------------------------------------------------

    def upsert_agenda_items(self, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not items:
            return []
        response = (
            self.client.table("agenda_items")
            .upsert(items, returning="representation")
            .execute()
        )
        return getattr(response, "data", []) or []

    def upsert_party_positions(self, positions: Iterable[Dict[str, Any]]) -> None:
        payload = self.ensure_serializable_items(positions)
        if not payload:
            return
        self.client.table("party_positions").upsert(payload).execute()

    def upsert_qa_interactions(self, interactions: Iterable[Dict[str, Any]]) -> None:
        payload = self.ensure_serializable_items(interactions)
        if not payload:
            return
        self.client.table("qa_interactions").upsert(payload).execute()

    def upsert_issue_trends(self, trends: Iterable[Dict[str, Any]]) -> None:
        payload = self.ensure_serializable_items(trends)
        if not payload:
            return
        self.client.table("issue_trends").upsert(payload).execute()

    def upsert_rag_documents(self, documents: Iterable[Dict[str, Any]]) -> None:
        payload = self.ensure_serializable_items(documents)
        if not payload:
            return
        self.client.table("documents_rag").upsert(payload).execute()

    def delete_rag_documents_by_source(self, *, source_id: str) -> None:
        self.client.table("documents_rag").delete().eq("source_id", source_id).execute()

    def delete_rag_documents_by_session(self, *, session_name: str) -> None:
        self.client.table("documents_rag").delete().eq(
            "metadata->>session_name", session_name
        ).execute()

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    def get_agenda_ids_for_session(self, session_id: str) -> List[str]:
        response = (
            self.client.table("agenda_items")
            .select("agenda_id")
            .eq("session_id", session_id)
            .execute()
        )
        return [row["agenda_id"] for row in getattr(response, "data", []) or []]

    def delete_party_positions_for_agendas(self, agenda_ids: Sequence[str]) -> None:
        if not agenda_ids:
            return
        self.client.table("party_positions").delete().in_("agenda_id", list(agenda_ids)).execute()

    def delete_qa_for_agendas(self, agenda_ids: Sequence[str]) -> None:
        if not agenda_ids:
            return
        self.client.table("qa_interactions").delete().in_("agenda_id", list(agenda_ids)).execute()

    def delete_agenda_items(self, session_id: str) -> None:
        self.client.table("agenda_items").delete().eq("session_id", session_id).execute()

    def delete_issue_trends(self, session_id: str) -> None:
        self.client.table("issue_trends").delete().eq("session_id", session_id).execute()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_serializable_items(items: Iterable[Any]) -> List[Dict[str, Any]]:
        """Normalize dataclasses or dict-like objects to plain dicts."""
        serialized: List[Dict[str, Any]] = []
        for item in items:
            if hasattr(item, "__dataclass_fields__"):
                serialized.append(asdict(item))
            elif isinstance(item, dict):
                serialized.append(item)
            else:
                raise TypeError(f"Unsupported payload type: {type(item)!r}")
        return serialized


