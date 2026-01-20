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
            print("  ⚠️  직렬화된 payload가 비어있습니다.")
            return
        
        print(f"  💾 documents_rag 테이블에 {len(payload)}개 문서 저장 중...")
        try:
            # 배치로 나누어 저장 (Supabase 제한 고려)
            batch_size = 100
            for i in range(0, len(payload), batch_size):
                batch = payload[i:i + batch_size]
                self.client.table("documents_rag").upsert(batch).execute()
                if (i // batch_size + 1) % 10 == 0:
                    print(f"    진행 중... {i + len(batch)}/{len(payload)}개 저장됨")
            print(f"  ✅ documents_rag 테이블 저장 완료: {len(payload)}개")
        except Exception as e:
            print(f"  ❌ documents_rag 저장 중 오류: {e}")
            import traceback
            traceback.print_exc()
            raise

    def delete_rag_documents_by_source(self, *, source_id: str) -> None:
        self.client.table("documents_rag").delete().eq("source_id", source_id).execute()

    def delete_rag_documents_by_session(self, *, session_name: str) -> None:
        """세션별 RAG 문서 삭제 (메타데이터 또는 source_id 패턴 기준)"""
        # 메타데이터 기준 삭제
        self.client.table("documents_rag").delete().eq(
            "metadata->>session_name", session_name
        ).execute()
        # source_id 패턴 기준 삭제 (이전 형식 데이터도 삭제하기 위해)
        # 1. 새 형식: session::{session_name}::로 시작
        # 2. 이전 형식: qa-XX (모든 qa_pair 타입 삭제 후 재저장)
        try:
            # 새 형식 삭제
            self.client.table("documents_rag").delete().ilike(
                "source_id", f"session::{session_name}::%"
            ).execute()
            self.client.table("documents_rag").delete().ilike(
                "source_id", f"{session_name}::%"
            ).execute()
            # 이전 형식 qa-XX 삭제
            # 방법 1: 메타데이터에 session_name이 있는 경우
            self.client.table("documents_rag").delete().eq(
                "source_type", "qa_pair"
            ).eq(
                "metadata->>session_name", session_name
            ).execute()
            # 방법 2: source_id가 qa-로 시작하고 메타데이터에 session_name이 없는 경우
            # 주의: 이것은 제415회 재저장 시에만 사용 (다른 세션 데이터는 보호됨)
            # 메타데이터가 null이거나 빈 값인 qa-XX 형식 삭제
            # 재저장 전이라면 이전 형식 데이터를 삭제해야 함
            try:
                # source_id가 qa-로 시작하고 메타데이터에 session_name이 null인 경우
                self.client.table("documents_rag").delete().eq(
                    "source_type", "qa_pair"
                ).like(
                    "source_id", "qa-%"
                ).or_(
                    f"metadata->>session_name.is.null,metadata->>session_name.eq."
                ).execute()
            except Exception:
                # or_ 구문이 작동하지 않을 수 있으므로, 일단 메타데이터 기준으로만 삭제
                pass
            # party_position도 동일하게 처리
            self.client.table("documents_rag").delete().eq(
                "source_type", "party_position"
            ).eq(
                "metadata->>session_name", session_name
            ).execute()
        except (AttributeError, Exception) as e:
            # ilike 메서드가 없는 경우 또는 다른 오류 발생 시
            # 메타데이터 기준으로만 삭제 (이미 위에서 처리됨)
            print(f"⚠️ source_id 패턴 삭제 중 오류 발생 (무시됨): {e}")

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
        import math
        import json
        
        def clean_value(val):
            """NaN, Inf 값을 None 또는 0으로 변환"""
            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    return 0.0
            elif isinstance(val, list):
                return [clean_value(v) for v in val]
            elif isinstance(val, dict):
                return {k: clean_value(v) for k, v in val.items()}
            return val
        
        serialized: List[Dict[str, Any]] = []
        for item in items:
            if hasattr(item, "__dataclass_fields__"):
                item_dict = asdict(item)
            elif isinstance(item, dict):
                item_dict = item
            else:
                raise TypeError(f"Unsupported payload type: {type(item)!r}")
            
            # NaN/Inf 값 정리
            cleaned_dict = clean_value(item_dict)
            serialized.append(cleaned_dict)
        
        return serialized


