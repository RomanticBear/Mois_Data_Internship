#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기본 모듈 테스트 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "analysis_scripts"))

def test_imports():
    """모듈 import 테스트"""
    print("=" * 60)
    print("모듈 Import 테스트")
    print("=" * 60)
    
    try:
        from parliament_analysis.analysis.models import (
            SessionSummary,
            PartyPosition,
            AgendaPartyAnalysis,
            QAAnalysisMetrics,
        )
        print("✅ analysis.models import 성공")
    except Exception as e:
        print(f"❌ analysis.models import 실패: {e}")
        return False
    
    try:
        from parliament_analysis.analysis.openai_analyzer import OpenAISessionAnalyzer
        print("✅ analysis.openai_analyzer import 성공")
    except Exception as e:
        print(f"❌ analysis.openai_analyzer import 실패: {e}")
        return False
    
    try:
        from parliament_analysis.pipeline.workflow import SessionAnalysisWorkflow
        print("✅ pipeline.workflow import 성공")
    except Exception as e:
        print(f"❌ pipeline.workflow import 실패: {e}")
        return False
    
    try:
        from parliament_analysis.data.embedding_client import EmbeddingClient
        print("✅ data.embedding_client import 성공")
    except Exception as e:
        print(f"❌ data.embedding_client import 실패: {e}")
        return False
    
    try:
        from parliament_analysis.rag.chunker import RAGChunker
        print("✅ rag.chunker import 성공")
    except Exception as e:
        print(f"❌ rag.chunker import 실패: {e}")
        return False
    
    return True


def test_models():
    """데이터 모델 생성 테스트"""
    print("\n" + "=" * 60)
    print("데이터 모델 생성 테스트")
    print("=" * 60)
    
    try:
        from parliament_analysis.analysis.models import SessionSummary
        
        summary = SessionSummary(
            session_name="제415회",
            meeting_date=None,
            key_issues=[
                {"issue": "재난안전", "importance": "높음", "description": "재난안전 대책 논의"}
            ],
            overall_sentiment=None,
            raw_summary="테스트 요약",
            metadata={}
        )
        print(f"✅ SessionSummary 생성 성공: {summary.session_name}")
        print(f"   - 핵심 이슈 수: {len(summary.key_issues)}")
    except Exception as e:
        print(f"❌ SessionSummary 생성 실패: {e}")
        return False
    
    return True


def test_chunker():
    """청킹 기능 테스트"""
    print("\n" + "=" * 60)
    print("청킹 기능 테스트")
    print("=" * 60)
    
    try:
        from parliament_analysis.rag.chunker import RAGChunker
        
        chunker = RAGChunker(chunk_size=100, overlap=20)
        
        test_text = "이것은 테스트 텍스트입니다. " * 10  # 긴 텍스트
        chunks = chunker._split_text(test_text)
        
        print(f"✅ 청킹 성공: {len(chunks)}개 청크 생성")
        print(f"   - 첫 번째 청크 길이: {len(chunks[0])}자")
    except Exception as e:
        print(f"❌ 청킹 실패: {e}")
        return False
    
    return True


def test_workflow_data_loading():
    """워크플로우 데이터 로딩 테스트 (실제 데이터 필요)"""
    print("\n" + "=" * 60)
    print("워크플로우 데이터 로딩 테스트")
    print("=" * 60)
    
    try:
        from parliament_analysis.pipeline.workflow import SessionAnalysisWorkflow
        
        # OpenAI 클라이언트 없이도 데이터 로딩은 가능
        workflow = SessionAnalysisWorkflow(openai_client=None, model="gpt-4o-mini")
        
        # 데이터 디렉토리 확인
        data_root = Path(__file__).resolve().parents[3] / "data" / "with_party"
        session_dir = data_root / "제415회"
        
        if session_dir.exists():
            print(f"✅ 데이터 디렉토리 존재: {session_dir}")
            print(f"   - CSV 파일 확인 가능")
        else:
            print(f"⚠️  데이터 디렉토리 없음: {session_dir}")
            print(f"   - 실제 데이터 로딩 테스트는 스킵")
    except Exception as e:
        print(f"❌ 워크플로우 테스트 실패: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("\n🧪 Parliament Analysis 모듈 테스트 시작\n")
    
    results = []
    results.append(("Import 테스트", test_imports()))
    results.append(("모델 생성 테스트", test_models()))
    results.append(("청킹 테스트", test_chunker()))
    results.append(("워크플로우 테스트", test_workflow_data_loading()))
    
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 모든 기본 테스트 통과!")
        print("\n💡 다음 단계:")
        print("   1. .env 파일에 OPENAI_API_KEY 설정")
        print("   2. python run_session_analysis.py 실행")
        print("   3. 실제 분석 테스트 수행")
    else:
        print("\n⚠️  일부 테스트 실패. 코드를 확인해주세요.")
    
    sys.exit(0 if all_passed else 1)




