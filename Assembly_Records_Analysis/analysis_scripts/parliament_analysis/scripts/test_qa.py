"""
RAG QA 시스템 테스트 스크립트
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[3]
sys.path.insert(0, str(project_root))
# analysis_scripts 경로 추가 (parliament_analysis 모듈 접근용)
analysis_scripts_path = current_dir.parents[1]
sys.path.insert(0, str(analysis_scripts_path))

from dotenv import load_dotenv

# .env 파일 로드 (Assembly_Records_Analysis 폴더에서)
env_path = project_root / "Assembly_Records_Analysis" / ".env"
if not env_path.exists():
    # 대체 경로 시도
    env_path = project_root / ".env"
load_dotenv(env_path)

from openai import OpenAI

from parliament_analysis.data.db_client import SupabaseDBClient
from parliament_analysis.data.embedding_client import EmbeddingClient
from parliament_analysis.rag.qa_system import RAGQASystem
from parliament_analysis.rag.retriever import RAGRetriever


def initialize_qa_system(session_name: str = None):
    """QA 시스템 초기화"""
    print("RAG QA 시스템 초기화")
    print("-" * 70)
    
    # 1. OpenAI 클라이언트 생성
    print("1. OpenAI 클라이언트 생성 중...", end=" ")
    llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedding_client = EmbeddingClient(openai_client=llm_client)
    print("완료")
    
    # 2. Supabase 클라이언트 생성
    print("2. Supabase 클라이언트 생성 중...", end=" ")
    db_client = SupabaseDBClient.from_env()
    print("완료")
    
    # 3. Embedding 클라이언트 생성
    print("3. Embedding 클라이언트 생성 중...", end=" ")
    # 이미 생성됨
    print("완료")
    
    # 4. Vector Store 생성
    print("4. Vector Store 생성 중...", end=" ")
    # Supabase에서 직접 사용
    print("완료")
    
    # 5. Retriever 생성
    print("5. Retriever 생성 중...", end=" ")
    retriever = RAGRetriever(
        db_client=db_client,
        embedding_client=embedding_client,
        verbose=False,  # 서버 측 검색 폴백 경고 메시지 숨김
    )
    print("완료")
    
    # 6. QA 시스템 생성
    print("6. QA 시스템 생성 중...", end=" ")
    qa_system = RAGQASystem(
        retriever=retriever,
        llm_client=llm_client,
    )
    print("완료")
    
    print("-" * 70)
    print("초기화 완료!")
    print()
    
    return qa_system


def test_qa_single(question: str, session_name: str = None, qa_system: RAGQASystem = None):
    """단일 질문 테스트"""
    if qa_system is None:
        qa_system = initialize_qa_system(session_name)
    
    print("="*70)
    print("질문:")
    print("="*70)
    print(question)
    print()
    
    # 질문 처리
    result = qa_system.ask_question(
        question,
        session_name=session_name,
        top_k=3,
    )
    
    # 답변 출력
    print("="*70)
    print("답변")
    print("="*70)
    answer = result.get("answer", "답변 없음")
    print(answer)
    
    # 참고 문서 상세 정보
    if result.get("sources"):
        print(f"\n{'='*70}")
        print(f"참고 문서: {len(result['sources'])}개")
        print("="*70)
        
        # 회차 필터링 검증
        if session_name:
            mismatched_count = 0
            for source in result["sources"]:
                metadata = source.get("metadata", {})
                doc_session = metadata.get("session_name")
                if doc_session and doc_session != session_name:
                    mismatched_count += 1
            
            if mismatched_count > 0:
                print(f"⚠️  경고: {mismatched_count}개 문서가 {session_name}가 아닙니다!")
        
        # 유사도가 높은 상위 2개 문서만 상세 표시
        for idx, source in enumerate(result["sources"][:2], 1):
            metadata = source.get("metadata", {})
            session = metadata.get("session_name", "N/A")
            source_type = source.get("source_type", "N/A")
            similarity = source.get("similarity", "N/A")
            content = source.get("content", "")
            
            # 회차 불일치 표시
            session_marker = ""
            if session_name and session != "N/A" and session != session_name:
                session_marker = " ⚠️ 불일치"
            
            print(f"\n[{idx}] {session}{session_marker}")
            print(f"    타입: {source_type}")
            if similarity != "N/A":
                print(f"    유사도: {similarity:.3f}")
            
            # 문서 내용 일부 표시 (최대 200자)
            if content:
                content_preview = content.strip()
                if len(content_preview) > 200:
                    content_preview = content_preview[:200] + "..."
                print(f"    내용: {content_preview}")
        
        # 나머지 문서는 간단히만 표시
        if len(result["sources"]) > 2:
            print(f"\n[3~{len(result['sources'])}] (생략)")
            for idx, source in enumerate(result["sources"][2:], 3):
                metadata = source.get("metadata", {})
                session = metadata.get("session_name", "N/A")
                source_type = source.get("source_type", "N/A")
                similarity = source.get("similarity", "N/A")
                session_marker = ""
                if session_name and session != "N/A" and session != session_name:
                    session_marker = " ⚠️"
                print(f"    [{idx}] {session}{session_marker} - {source_type} (유사도: {similarity:.3f})")
    
    print("\n" + "="*70)
    
    return result


def test_qa_interactive(session_name: str = None):
    """인터랙티브 모드"""
    print("인터랙티브 모드 시작")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("="*70)
    print()
    
    # QA 시스템 초기화 (한 번만)
    qa_system = initialize_qa_system(session_name)
    
    while True:
        try:
            question = input("질문: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("종료합니다.")
                break
            
            print()
            test_qa_single(question, session_name=session_name, qa_system=qa_system)
            print()
            
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()
            print()


def test_qa_comprehensive(session_name: str = None):
    """종합 테스트 - 모든 질문 타입 테스트"""
    print("=" * 70)
    print("🧪 RAG QA 시스템 종합 테스트")
    print("=" * 70)
    print()
    
    qa_system = initialize_qa_system(session_name)
    
    # 테스트 질문 리스트
    test_questions = [
        # 통계 질문
        ("통계", "제415회에서 몇 건의 질의가 있었나요?", session_name or "제415회"),
        ("통계", "제415회의 안건은 몇 건인가요?", session_name or "제415회"),
        ("통계", "제415회에서 더불어민주당의 QA는 몇 건인가요?", session_name or "제415회"),
        ("통계", "제415회에서 가장 활발한 의원은?", session_name or "제415회"),
        
        # 비교 질문
        ("비교", "제415회와 제417회의 차이는?", None),
        ("비교", "재난안전 예산 안건에서 정당 간 합의점은?", None),
        ("비교", "더불어민주당의 입장은?", None),
        
        # 시계열 질문
        ("시계열", "재난안전 예산 안건의 진행 추이를 알려줘", None),
        ("시계열", "안전 이슈의 트렌드를 분석해줘", None),
        
        # 검색 질문
        ("검색", "제415회에서 논의된 주요 이슈는?", session_name or "제415회"),
        ("검색", "더불어민주당의 주요 관심사는?", None),
        
        # 보고서 질문
        ("보고서", "제415회 세션 요약 보고서 생성해줘", session_name or "제415회"),
    ]
    
    results = []
    for category, question, session in test_questions:
        print(f"\n{'=' * 70}")
        print(f"[{category}] {question}")
        print("=" * 70)
        try:
            result = qa_system.ask_question(
                question,
                session_name=session,
                top_k=3,
            )
            
            print(f"\n답변:")
            print(result.get("answer", "답변 없음"))
            print(f"\n질문 타입: {result.get('question_type', 'N/A')}")
            
            if result.get("statistics"):
                print(f"통계 데이터: {result['statistics']}")
            if result.get("comparison_data"):
                print(f"비교 데이터: (있음)")
            if result.get("timeseries_data"):
                print(f"시계열 데이터: (있음)")
            if result.get("report_data"):
                print(f"보고서 데이터: (있음)")
            
            results.append({
                "category": category,
                "question": question,
                "success": True,
                "question_type": result.get("question_type"),
            })
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "category": category,
                "question": question,
                "success": False,
                "error": str(e),
            })
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 테스트 결과 요약")
    print("=" * 70)
    
    success_count = sum(1 for r in results if r.get("success"))
    total_count = len(results)
    
    print(f"\n총 {total_count}개 질문 중 {success_count}개 성공 ({success_count/total_count*100:.1f}%)")
    
    print("\n카테고리별 결과:")
    for category in ["통계", "비교", "시계열", "검색", "보고서"]:
        category_results = [r for r in results if r.get("category") == category]
        category_success = sum(1 for r in category_results if r.get("success"))
        print(f"  {category}: {category_success}/{len(category_results)} 성공")
    
    print("\n실패한 질문:")
    for r in results:
        if not r.get("success"):
            print(f"  - [{r.get('category')}] {r.get('question')}")
            print(f"    오류: {r.get('error', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="RAG QA 시스템 테스트")
    parser.add_argument(
        "--question",
        type=str,
        help="질문 텍스트",
    )
    parser.add_argument(
        "--session",
        type=str,
        help="회차 이름 (예: 제415회)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="인터랙티브 모드",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="모든 질문 타입 종합 테스트",
    )
    
    args = parser.parse_args()
    
    if args.test_all:
        test_qa_comprehensive(session_name=args.session)
    elif args.interactive:
        test_qa_interactive(session_name=args.session)
    elif args.question:
        test_qa_single(args.question, session_name=args.session)
    else:
        # 기본: 종합 테스트
        test_qa_comprehensive(session_name=args.session)


if __name__ == "__main__":
    main()

