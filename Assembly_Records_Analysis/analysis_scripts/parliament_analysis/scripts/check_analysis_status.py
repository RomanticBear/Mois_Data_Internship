#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
분석 진행 상황 확인 스크립트
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "analysis_scripts"))

from parliament_analysis.data.db_client import SupabaseDBClient

load_dotenv()


def check_analysis_status(session_names: list = None):
    """분석 진행 상황 확인"""
    if session_names is None:
        session_names = ["제415회", "제416회", "제417회"]
    
    print("=" * 70)
    print("📊 분석 진행 상황 확인")
    print("=" * 70)
    
    if not os.getenv("SUPABASE_URL") or not (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_API_KEY")
    ):
        print("❌ SUPABASE_URL 및 SUPABASE_SERVICE_ROLE_KEY 환경 변수를 설정해주세요.")
        return
    
    db_client = SupabaseDBClient.from_env()
    
    for session_name in session_names:
        print(f"\n📌 {session_name}:")
        
        # 세션 존재 여부
        session = db_client.get_session_record(session_name)
        if session:
            print(f"  ✅ 세션 레코드 존재")
            print(f"     분석일시: {session.get('analyzed_at', 'N/A')}")
        else:
            print(f"  ⚠️  세션 레코드 없음 (아직 분석 안됨)")
            continue
        
        # RAG 문서 수
        docs = db_client.client.table("documents_rag").select("document_id", count="exact").eq("metadata->>session_name", session_name).execute()
        doc_count = docs.count if hasattr(docs, "count") else len(docs.data or [])
        print(f"  📚 RAG 문서: {doc_count}개")
        
        # 안건 수
        if session:
            session_id = session.get("session_id")
            agenda_ids = db_client.get_agenda_ids_for_session(session_id)
            print(f"  📋 안건 수: {len(agenda_ids)}개")
            
            # QA 수
            if agenda_ids:
                qa_count = 0
                for agenda_id in agenda_ids:
                    qas = db_client.client.table("qa_interactions").select("qa_id", count="exact").eq("agenda_id", agenda_id).execute()
                    qa_count += (qas.count if hasattr(qas, "count") else len(qas.data or []))
                print(f"  💬 QA 수: {qa_count}개")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="분석 진행 상황 확인")
    parser.add_argument(
        "--sessions",
        nargs="+",
        help="확인할 회차 목록",
    )
    
    args = parser.parse_args()
    check_analysis_status(args.sessions)


