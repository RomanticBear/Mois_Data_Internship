#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국회 회의록 OpenAI 기반 심층 분석 (RAG 파이프라인 통합)
"""

from __future__ import annotations

import os
import warnings
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from ..data.db_client import SupabaseDBClient
from ..data.embedding_client import EmbeddingClient
from ..pipeline.utils import generate_analysis_version
from ..pipeline.workflow import SessionAnalysisWorkflow
from ..pipeline.persistence import persist_analysis_to_supabase
from ..rag.chunker import RAGChunker
from ..rag.vector_store import VectorStore

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

load_dotenv()


def create_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)


def ensure_supabase_clients(
    openai_client: OpenAI,
) -> Optional[Tuple[SupabaseDBClient, EmbeddingClient, VectorStore, RAGChunker]]:
    if not os.getenv("SUPABASE_URL") or not (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_API_KEY")
    ):
        return None

    db_client = SupabaseDBClient.from_env()
    embedding_client = EmbeddingClient(openai_client=openai_client)
    vector_store = VectorStore(db_client=db_client, embedding_client=embedding_client)
    chunker = RAGChunker()
    return db_client, embedding_client, vector_store, chunker


def create_visualizations(results: Dict[str, Any], session_name: str) -> None:
    """분석 결과 시각화."""
    print("\n📊 시각화 생성 중...")

    output_dir = "outputs/analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    summary = results.get("session_summary")
    if summary and summary.get("key_issues"):
        fig, ax = plt.subplots(figsize=(12, 6))
        issues = [issue["issue"] for issue in summary["key_issues"]]
        importance_map = {"높음": 3, "중간": 2, "낮음": 1}
        importance_scores = [
            importance_map.get(issue.get("importance", "중간"), 2)
            for issue in summary["key_issues"]
        ]

        ax.barh(issues, importance_scores, color="steelblue")
        ax.set_xlabel("중요도", fontsize=12)
        ax.set_title(f"{session_name} 핵심 이슈 중요도", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 4)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{session_name}_key_issues.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("  ✅ 핵심 이슈 중요도 차트 생성")

    party_overview = summary.get("metadata", {}).get("party_positions_overview") if summary else None
    if party_overview:
        parties = list(party_overview.keys())
        if parties:
            fig, ax = plt.subplots(figsize=(10, 6))
            concerns_count = [
                len(party_overview[party].get("main_concerns", [])) for party in parties
            ]

            ax.bar(parties, concerns_count, color="coral")
            ax.set_ylabel("주요 관심사 수", fontsize=12)
            ax.set_title(
                f"{session_name} 정당별 주요 관심사 (질의-응답 회의)",
                fontsize=14,
                fontweight="bold",
            )
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{session_name}_party_concerns.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print("  ✅ 정당별 관심사 차트 생성")

    qa = results.get("qa_analysis")
    if qa and qa.get("quality_distribution"):
        def _to_numeric(value: Any) -> float:
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                sanitized = value.strip().replace("%", "")
                if not sanitized:
                    return 0.0
                try:
                    return float(sanitized)
                except ValueError:
                    return 0.0
            return 0.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        quality = qa["quality_distribution"]
        labels = ["고품질", "중품질", "저품질"]
        sizes = [
            _to_numeric(quality.get("high")),
            _to_numeric(quality.get("medium")),
            _to_numeric(quality.get("low")),
        ]
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]

        ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax1.set_title("응답 품질 분포", fontsize=12, fontweight="bold")

        q_types = qa.get("question_types", {})
        if q_types:
            type_labels = list(q_types.keys())
            type_values = [_to_numeric(q_types.get(k)) for k in type_labels]

            ax2.bar(range(len(type_labels)), type_values, color="steelblue")
            ax2.set_xticks(range(len(type_labels)))
            ax2.set_xticklabels(["정책 질의", "사실 확인", "비판 질의", "제안 질의"], rotation=45, ha="right")
            ax2.set_ylabel("비율 (%)", fontsize=12)
            ax2.set_title("질문 유형 분포", fontsize=12, fontweight="bold")

        plt.suptitle(f"{session_name} 질의-응답 효과성 분석", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{session_name}_qa_quality.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("  ✅ 질의-응답 품질 차트 생성")

    print(f"📊 시각화 완료: analysis_results/{session_name}_*.png")


def main(session_name: str = "제415회") -> None:
    """메인 실행 함수."""
    print("=" * 60)
    print(f"{session_name} 국회 회의록 OpenAI 심층 분석")
    print("=" * 60)

    openai_client = create_openai_client()
    workflow = SessionAnalysisWorkflow(openai_client=openai_client)

    print(f"📊 {session_name} 데이터 로딩 중...")
    df = workflow.load_session_data(session_name=session_name)
    print(f"✅ 총 {len(df):,}개 발언 로드 완료")

    print("\n🔍 최소 필터링 수행 중 (문맥 판단은 OpenAI가 수행)...")
    quality_df = workflow.filter_quality_speeches(df)
    print("📊 필터링 결과:")
    print(f"  - 전체 발언: {len(df):,}개")
    print(f"  - 유효 발언: {len(quality_df):,}개 ({len(quality_df)/len(df)*100:.1f}%)")
    print("  ⚠️ 키워드 기반 필터링 제거: 문맥 판단은 OpenAI가 수행합니다")

    hash_digest = workflow.compute_dataframe_hash(df)
    analysis_version = generate_analysis_version()

    results: Dict[str, Any] = {
        "session_name": session_name,
        "total_speeches": len(df),
        "quality_speeches": len(quality_df),
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "analysis_version": analysis_version,
        "hash_digest": hash_digest,
    }

    print("\n" + "=" * 60)
    print("1단계: 회차별 핵심 이슈 요약")
    print("=" * 60)
    summary_payload = workflow.prepare_session_summary_payload(quality_df)
    session_summary = workflow.run_session_summary(session_name, payload=summary_payload)
    if session_summary:
        results["session_summary"] = asdict(session_summary)

    agenda_payloads = workflow.prepare_agenda_payloads(quality_df, top_agendas=3)
    party_analyses = workflow.run_party_positions(session_name, agenda_payloads=agenda_payloads)
    if party_analyses:
        results["party_positions"] = [asdict(analysis) for analysis in party_analyses]

    qa_pairs = workflow.prepare_qa_pairs(quality_df, session_name)
    if not qa_pairs:
        print("  ⚠️ 질의-응답 쌍을 찾을 수 없습니다.")
    qa_metrics = workflow.run_qa_analysis(session_name, qa_pairs=qa_pairs)
    if qa_metrics:
        results["qa_analysis"] = asdict(qa_metrics)

    output_dir = "outputs/analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{session_name}_openai_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 분석 결과 저장: {json_path}")
    create_visualizations(results, session_name)

    supabase_clients = ensure_supabase_clients(openai_client)
    if supabase_clients:
        db_client, embedding_client, vector_store, chunker = supabase_clients
        print("\n☁️ Supabase에 분석 결과 동기화 중...")
        persist_analysis_to_supabase(
            session_name=session_name,
            hash_digest=hash_digest,
            analysis_version=analysis_version,
            raw_df=df,
            quality_df=quality_df,
            session_summary=session_summary,
            party_analyses=party_analyses,
            qa_pairs=qa_pairs,
            qa_metrics=qa_metrics,
            db_client=db_client,
            embedding_client=embedding_client,
            vector_store=vector_store,
            chunker=chunker,
        )
        print("✅ Supabase 동기화 완료")
    else:
        print("\nℹ️ Supabase 환경 변수가 설정되지 않아 로컬 JSON/시각화만 생성했습니다.")

    print("\n" + "=" * 60)
    print("분석 완료 요약")
    print("=" * 60)
    print(f"📁 결과 파일: {json_path}")
    print(f"📊 총 발언 수: {len(df):,}개")
    print(f"✅ 품질 발언 수: {len(quality_df):,}개")

    if session_summary and session_summary.key_issues:
        print(f"\n🔍 핵심 이슈: {len(session_summary.key_issues)}개")
        for issue in session_summary.key_issues[:3]:
            print(f"  - {issue.get('issue', 'N/A')} ({issue.get('importance', 'N/A')})")

    if party_analyses:
        print(f"\n📋 분석된 안건 수: {len(party_analyses)}개")

    if qa_metrics:
        print(f"\n💬 질의-응답 쌍: {qa_metrics.total_qa_pairs}개")
        if qa_metrics.quality_distribution:
            print(f"  - 고품질 응답: {qa_metrics.quality_distribution.get('high', 0)}%")

    print("\n✅ 모든 분석이 완료되었습니다!")


if __name__ == "__main__":
    main()

