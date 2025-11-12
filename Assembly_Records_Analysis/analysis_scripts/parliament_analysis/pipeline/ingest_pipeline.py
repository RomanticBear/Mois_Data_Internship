"""
[역할] 배치 처리 파이프라인 (여러 세션 일괄 처리)
- IngestPipeline.run(): 여러 세션 일괄 처리
- _filter_sessions_to_process(): 처리할 세션 필터링 (중복 체크)
- _process_single_session(): 단일 세션 처리
- 세션 해시 및 분석 버전 비교를 통한 중복 처리 방지
- force_recompute 옵션으로 강제 재실행 지원
- 현재 run_session_analysis.py에서 직접 사용하지 않음 (향후 배치 처리용)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ..data.db_client import SupabaseDBClient
from ..data.embedding_client import EmbeddingClient
from ..pipeline.persistence import persist_analysis_to_supabase
from ..pipeline.workflow import SessionAnalysisWorkflow
from ..rag.chunker import RAGChunker
from ..rag.vector_store import VectorStore
from .utils import PipelineContext, generate_analysis_version


class IngestPipeline:
    """Coordinate new session detection, analysis, and storage."""

    def __init__(
        self,
        *,
        workflow: SessionAnalysisWorkflow,
        db_client: SupabaseDBClient,
        embedding_client: EmbeddingClient,
        chunker: Optional[RAGChunker] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self.workflow = workflow
        self.db_client = db_client
        self.embedding_client = embedding_client
        self.chunker = chunker or RAGChunker()
        self.vector_store = vector_store or VectorStore(
            db_client=db_client, embedding_client=embedding_client
        )

    def run(
        self,
        *,
        sessions: Sequence[dict],
        force_recompute: bool = False,
        context: Optional[PipelineContext] = None,
    ) -> None:
        """Main entry point for new session analyses."""
        if not sessions:
            return

        analysis_version = generate_analysis_version()

        enriched_sessions: List[Dict[str, object]] = []
        for session_payload in sessions:
            session_name = session_payload["session_name"]
            df: pd.DataFrame = session_payload.get("dataframe")  # type: ignore[assignment]
            if df is None:
                df = self.workflow.load_session_data(session_name=session_name)
            hash_digest = self.workflow.compute_dataframe_hash(df)

            enriched_sessions.append(
                {
                    "session_name": session_name,
                    "dataframe": df,
                    "hash_digest": hash_digest,
                    "config": session_payload,
                }
            )

        sessions_to_process = list(
            self._filter_sessions_to_process(
                enriched_sessions, version=analysis_version, force=force_recompute
            )
        )

        for session_entry in sessions_to_process:
            self._process_single_session(session_entry, version=analysis_version)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_sessions_to_process(
        self,
        sessions: Sequence[Dict[str, object]],
        *,
        version: str,
        force: bool,
    ) -> Iterable[Dict[str, object]]:
        if force:
            return sessions

        metadata = [
            {
                "session_name": entry["session_name"],
                "hash_digest": entry["hash_digest"],
            }
            for entry in sessions
        ]

        to_process_metadata = self.db_client.get_sessions_to_process(
            session_metadata=metadata,
            analysis_version=version,
            force=force,
        )
        names_to_process = {item["session_name"] for item in to_process_metadata}
        return [entry for entry in sessions if entry["session_name"] in names_to_process]

    def _process_single_session(self, session_entry: Dict[str, object], *, version: str) -> None:
        session_name: str = session_entry["session_name"]  # type: ignore[assignment]
        df: pd.DataFrame = session_entry["dataframe"]  # type: ignore[assignment]
        hash_digest: str = session_entry["hash_digest"]  # type: ignore[assignment]

        quality_df = self.workflow.filter_quality_speeches(df)

        summary_payload = self.workflow.prepare_session_summary_payload(quality_df)
        session_summary = self.workflow.run_session_summary(
            session_name, payload=summary_payload
        )

        agenda_payloads = self.workflow.prepare_agenda_payloads(quality_df, top_agendas=3)
        party_analyses = self.workflow.run_party_positions(
            session_name, agenda_payloads=agenda_payloads
        )

        qa_pairs = self.workflow.prepare_qa_pairs(quality_df, session_name)
        qa_metrics = self.workflow.run_qa_analysis(
            session_name, qa_pairs=qa_pairs
        )

        self._persist_results(
            session_name=session_name,
            analysis_results={
                "hash_digest": hash_digest,
                "analysis_version": version,
                "raw_df": df,
                "quality_df": quality_df,
                "session_summary": session_summary,
                "party_analyses": party_analyses,
                "qa_pairs": qa_pairs,
                "qa_metrics": qa_metrics,
            },
        )

    def _persist_results(self, *, session_name: str, analysis_results: dict) -> None:
        persist_analysis_to_supabase(
            session_name=session_name,
            hash_digest=analysis_results["hash_digest"],
            analysis_version=analysis_results["analysis_version"],
            raw_df=analysis_results["raw_df"],
            quality_df=analysis_results["quality_df"],
            session_summary=analysis_results["session_summary"],
            party_analyses=analysis_results["party_analyses"],
            qa_pairs=analysis_results["qa_pairs"],
            qa_metrics=analysis_results["qa_metrics"],
            db_client=self.db_client,
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
            chunker=self.chunker,
        )
