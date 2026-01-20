#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
특정 회차의 Supabase 데이터 삭제 스크립트
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


def delete_session_data(session_name: str) -> None:
    """특정 회차의 모든 데이터 삭제"""
    print("=" * 70)
    print(f"🗑️  {session_name} 회차 데이터 삭제")
    print("=" * 70)
    
    if not os.getenv("SUPABASE_URL") or not (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_API_KEY")
    ):
        print("❌ SUPABASE_URL 및 SUPABASE_SERVICE_ROLE_KEY 환경 변수를 설정해주세요.")
        return
    
    db_client = SupabaseDBClient.from_env()
    
    # 1. 세션 레코드 조회
    session_record = db_client.get_session_record(session_name)
    if not session_record:
        print(f"⚠️  {session_name} 회차 데이터를 찾을 수 없습니다.")
        return
    
    session_id = session_record.get("session_id")
    print(f"📌 세션 ID: {session_id}")
    
    # 2. 관련 데이터 삭제 (외래키 제약을 고려한 순서)
    print("\n🗑️  관련 데이터 삭제 중...")
    
    # RAG 문서 삭제
    print("  - RAG 문서 삭제 중...")
    db_client.delete_rag_documents_by_session(session_name=session_name)
    print("    ✅ RAG 문서 삭제 완료")
    
    if session_id:
        # agenda_id 먼저 조회
        agenda_ids = db_client.get_agenda_ids_for_session(session_id)
        
        # 외래키 제약을 고려하여 역순으로 삭제
        if agenda_ids:
            print("  - 정당 입장 삭제 중...")
            db_client.delete_party_positions_for_agendas(agenda_ids)
            print("    ✅ 정당 입장 삭제 완료")
            
            print("  - QA 상호작용 삭제 중...")
            db_client.delete_qa_for_agendas(agenda_ids)
            print("    ✅ QA 상호작용 삭제 완료")
        
        # 안건 아이템 삭제 (정당 입장, QA 삭제 후)
        print("  - 안건 아이템 삭제 중...")
        db_client.delete_agenda_items(session_id)
        print("    ✅ 안건 아이템 삭제 완료")
        
        # 이슈 트렌드 삭제
        print("  - 이슈 트렌드 삭제 중...")
        db_client.delete_issue_trends(session_id)
        print("    ✅ 이슈 트렌드 삭제 완료")
    
    # 3. 세션 레코드 삭제
    print("  - 세션 레코드 삭제 중...")
    db_client.client.table("sessions").delete().eq("session_id", session_id).execute()
    print("    ✅ 세션 레코드 삭제 완료")
    
    print("\n" + "=" * 70)
    print(f"✅ {session_name} 회차 데이터 삭제 완료")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="특정 회차의 Supabase 데이터 삭제")
    parser.add_argument(
        "session_name",
        type=str,
        help="삭제할 회차 이름 (예: 제415회)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="확인 없이 바로 삭제",
    )
    
    args = parser.parse_args()
    
    if not args.confirm:
        response = input(f"⚠️  {args.session_name} 회차의 모든 데이터를 삭제하시겠습니까? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("❌ 삭제가 취소되었습니다.")
            sys.exit(0)
    
    delete_session_data(args.session_name)

