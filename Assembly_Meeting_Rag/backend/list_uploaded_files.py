"""
Vector Store에 업로드된 파일 목록 출력
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "backend"))

from app.services.vector_store import VectorStoreService
from app.services.metadata_db import MetadataDBService

load_dotenv()

def list_uploaded_files():
    """업로드된 파일 목록 출력"""
    print("=" * 80)
    print("Vector Store에 업로드된 파일 목록")
    print("=" * 80)
    print()
    
    # 서비스 초기화
    metadata_db = MetadataDBService()
    vector_store = VectorStoreService(metadata_db=metadata_db)
    
    # Vector Store 정보
    import requests
    vs_url = f"{vector_store.base_url}/vector_stores/{vector_store.vector_store_id}"
    vs_info_response = requests.get(vs_url, headers=vector_store.headers)
    
    if vs_info_response.status_code == 200:
        vs_info = vs_info_response.json()
        file_counts = vs_info.get("file_counts", {})
        print(f"📊 Vector Store ID: {vector_store.vector_store_id}")
        print(f"📈 파일 통계:")
        print(f"   - 총 파일 수: {file_counts.get('total', 0)}개")
        print(f"   - 완료: {file_counts.get('completed', 0)}개")
        print(f"   - 진행중: {file_counts.get('in_progress', 0)}개")
        print(f"   - 실패: {file_counts.get('failed', 0)}개")
        print()
    
    # 메타DB에서 업로드된 파일 조회
    all_docs = metadata_db.get_all_documents()
    uploaded_docs = [doc for doc in all_docs if doc.vector_store_file_id]
    
    # 날짜순 정렬
    uploaded_docs.sort(key=lambda x: (x.date, x.meeting_number))
    
    print(f"📋 업로드된 파일 목록 ({len(uploaded_docs)}개):")
    print()
    
    # 회차별로 그룹화
    by_session = {}
    for doc in uploaded_docs:
        session = doc.assembly_number
        if session not in by_session:
            by_session[session] = []
        by_session[session].append(doc)
    
    # 회차순 정렬
    sorted_sessions = sorted(by_session.keys(), key=lambda x: int(x.replace("제", "").replace("회", "")))
    
    total_count = 0
    for session in sorted_sessions:
        docs = by_session[session]
        docs.sort(key=lambda x: (x.date, x.meeting_number))
        
        print(f"📁 {session} ({len(docs)}개 파일)")
        for i, doc in enumerate(docs, 1):
            print(f"   {i:2d}. {doc.filename}")
            print(f"       - 위원회: {doc.committee}")
            print(f"       - 회의번호: {doc.meeting_number}차")
            print(f"       - 날짜: {doc.date}")
            print(f"       - File ID: {doc.vector_store_file_id}")
            print()
            total_count += 1
    
    print("=" * 80)
    print(f"✅ 총 {total_count}개 파일이 Vector Store에 업로드되어 있습니다.")
    print("=" * 80)

if __name__ == "__main__":
    list_uploaded_files()




