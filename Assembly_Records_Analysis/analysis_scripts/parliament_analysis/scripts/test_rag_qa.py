#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG QA 시스템 테스트 스크립트
Supabase에 저장된 벡터 데이터를 사용하여 질문-답변 테스트
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "analysis_scripts"))

from parliament_analysis.data.db_client import SupabaseDBClient
from parliament_analysis.data.embedding_client import EmbeddingClient
from parliament_analysis.rag.retriever import RAGRetriever
from parliament_analysis.rag.qa_system import RAGQASystem

load_dotenv()


def initialize_qa_system() -> RAGQASystem | None:
    """QA 시스템 초기화"""
    # OpenAI 클라이언트
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        return None
    openai_client = OpenAI(api_key=api_key)

    # Supabase 클라이언트
    if not os.getenv("SUPABASE_URL") or not (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_API_KEY")
    ):
        print("❌ SUPABASE_URL 및 SUPABASE_SERVICE_ROLE_KEY 환경 변수를 설정해주세요.")
        return None

    db_client = SupabaseDBClient.from_env()
    embedding_client = EmbeddingClient(openai_client=openai_client)
    retriever = RAGRetriever(
        db_client=db_client,
        embedding_client=embedding_client,
    )
    qa_system = RAGQASystem(
        retriever=retriever,
        llm_client=openai_client,
    )

    return qa_system


def check_database_status():
    """데이터베이스 상태 확인"""
    try:
        db_client = SupabaseDBClient.from_env()
        
        # 세션 수 확인
        sessions = db_client.client.table("sessions").select("session_name").execute()
        session_count = len(sessions.data) if sessions.data else 0
        
        # RAG 문서 수 확인 (더 안전한 방법)
        try:
            documents = db_client.client.table("documents_rag").select("*", count="exact").limit(1).execute()
            doc_count = documents.count if hasattr(documents, 'count') else 0
            # count가 없으면 실제 데이터로 확인
            if doc_count == 0:
                all_docs = db_client.client.table("documents_rag").select("*").limit(100).execute()
                doc_count = len(all_docs.data) if all_docs.data else 0
        except Exception as e:
            print(f"⚠️  RAG 문서 테이블 확인 실패: {e}")
            doc_count = 0
        
        return True, session_count, doc_count
    except Exception as e:
        print(f"⚠️  데이터베이스 연결 실패: {e}")
        return False, 0, 0


def run_quick_test(qa_system: RAGQASystem, session_name: str | None = None):
    """빠른 테스트 질문들"""
    test_questions = [
        "제415회 국회에서 논의된 주요 이슈는 무엇인가요?",
        "화성 공장 화재와 관련된 논의가 있었나요?",
        "정당별로 어떤 입장 차이가 있었나요?",
    ]

    print("=" * 70)
    print("🧪 RAG QA 시스템 빠른 테스트")
    print("=" * 70)
    if session_name:
        print(f"📌 {session_name} 회차만 검색")
    else:
        print("📌 전체 세션 검색")
    print()

    for i, question in enumerate(test_questions, 1):
        print("=" * 70)
        print(f"질문 {i}: {question}")
        print("=" * 70)
        
        try:
            result = qa_system.ask_question(question, session_name=session_name)
            
            print(f"\n💬 답변:")
            print(result.get("answer", "답변을 생성할 수 없습니다."))
            
            if result.get("sources"):
                print(f"\n📚 참조 문서 ({len(result['sources'])}개):")
                for j, source in enumerate(result["sources"][:3], 1):
                    print(f"\n  [{j}] {source.get('source_type', 'unknown')} (유사도: {source.get('similarity', 0):.3f})")
                    if source.get('metadata'):
                        meta = source['metadata']
                        if meta.get('session_name'):
                            print(f"      세션: {meta['session_name']}")
                        if meta.get('party_name'):
                            print(f"      정당: {meta['party_name']}")
                        if meta.get('agenda_title'):
                            print(f"      안건: {meta['agenda_title']}")
                    content_preview = source.get('content', '')[:100]
                    if content_preview:
                        print(f"      내용: {content_preview}...")
            
            print()
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            print()

    print("=" * 70)
    print("✅ 빠른 테스트 완료")
    print("=" * 70)


def ask_question_interactive(qa_system: RAGQASystem, session_name: str | None = None):
    """인터랙티브 질문-답변"""
    print("=" * 70)
    print("💬 RAG QA 시스템 인터랙티브 모드")
    print("=" * 70)
    if session_name:
        print(f"📌 {session_name} 회차만 검색")
    else:
        print("📌 전체 세션 검색")
    print("\n질문을 입력하세요. 종료하려면 'quit' 또는 'exit'를 입력하세요.\n")

    while True:
        try:
            question = input("질문: ").strip()
            
            if question.lower() in ["quit", "exit", "종료", "q"]:
                print("\n👋 종료합니다.")
                break

            if not question:
                continue

            print("\n🔍 검색 중...")
            result = qa_system.ask_question(question, session_name=session_name)

            print(f"\n💬 답변:")
            print(result.get("answer", "답변을 생성할 수 없습니다."))

            if result.get("sources"):
                print(f"\n📚 참조 문서 ({len(result['sources'])}개):")
                for i, source in enumerate(result["sources"][:3], 1):
                    print(f"\n  [{i}] {source.get('source_type', 'unknown')} (유사도: {source.get('similarity', 0):.3f})")
                    if source.get('metadata'):
                        meta = source['metadata']
                        if meta.get('session_name'):
                            print(f"      세션: {meta['session_name']}")
                        if meta.get('party_name'):
                            print(f"      정당: {meta['party_name']}")

            print("\n" + "-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}\n")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG QA 시스템 테스트 스크립트")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="미리 정의된 질문들로 빠른 테스트 모드 실행",
    )
    parser.add_argument(
        "--session",
        type=str,
        help="특정 세션 이름으로 필터링 (예: 제415회)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 RAG QA 시스템 테스트")
    print("=" * 70)
    print()

    # 데이터베이스 상태 확인
    db_ok, session_count, doc_count = check_database_status()
    if not db_ok:
        print("⚠️  데이터베이스 연결 실패. Supabase 환경 변수를 확인하세요.")
        sys.exit(1)

    print(f"📊 데이터베이스 상태:")
    print(f"  - 세션 수: {session_count}개")
    print(f"  - RAG 문서 수: {doc_count}개")
    print()

    if doc_count == 0:
        print("⚠️  RAG 문서가 없습니다. 먼저 run_session_analysis.py를 실행하여 데이터를 저장하세요.")
        sys.exit(1)

    # QA 시스템 초기화
    qa_system = initialize_qa_system()
    if not qa_system:
        print("⚠️  QA 시스템 초기화 실패.")
        sys.exit(1)

    print("✅ QA 시스템 초기화 완료\n")

    session_name = args.session

    if args.quick:
        print("\n📋 빠른 테스트 모드로 실행합니다.\n")
        run_quick_test(qa_system, session_name=session_name)
    else:
        # 인터랙티브 질문-답변 시작
        ask_question_interactive(qa_system, session_name=session_name)


if __name__ == "__main__":
    main()

