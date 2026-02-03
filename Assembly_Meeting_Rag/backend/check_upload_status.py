"""
Vector Store 및 메타DB 업로드 상태 확인 스크립트
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from app.services.vector_store import VectorStoreService
from app.services.metadata_db import MetadataDBService

load_dotenv()

def check_vector_store_status(vector_store: VectorStoreService):
    """Vector Store 상태 확인"""
    print("=" * 80)
    print("📊 Vector Store 상태")
    print("=" * 80)
    print()
    
    if not vector_store.vector_store_id:
        print("❌ Vector Store가 초기화되지 않았습니다.")
        return
    
    print(f"Vector Store ID: {vector_store.vector_store_id}")
    print()
    
    # Vector Store 상세 정보
    vs_url = f"{vector_store.base_url}/vector_stores/{vector_store.vector_store_id}"
    response = requests.get(vs_url, headers=vector_store.headers)
    
    if response.status_code == 200:
        vs_data = response.json()
        file_counts = vs_data.get("file_counts", {})
        
        print("📈 파일 통계:")
        print(f"  - 총 파일 수: {file_counts.get('total', 0)}개")
        print(f"  - 완료: {file_counts.get('completed', 0)}개")
        print(f"  - 진행중: {file_counts.get('in_progress', 0)}개")
        print(f"  - 실패: {file_counts.get('failed', 0)}개")
        print()
        
        # Vector Store의 파일 목록
        files_url = f"{vector_store.base_url}/vector_stores/{vector_store.vector_store_id}/files"
        files_response = requests.get(files_url, headers=vector_store.headers, params={"limit": 100})
        
        if files_response.status_code == 200:
            files_data = files_response.json()
            files = files_data.get("data", [])
            
            if files:
                print(f"📋 Vector Store 파일 목록 ({len(files)}개):")
                for i, file_info in enumerate(files, 1):
                    status = file_info.get("status", "unknown")
                    file_id = file_info.get("id", "N/A")
                    created_at = file_info.get("created_at", "N/A")
                    print(f"  {i}. File ID: {file_id}")
                    print(f"     상태: {status}")
                    print(f"     생성일: {created_at}")
                    print()
            else:
                print("⚠️ Vector Store에 파일이 없습니다.")
        else:
            print(f"⚠️ 파일 목록 조회 실패: {files_response.status_code}")
    else:
        print(f"❌ Vector Store 정보 조회 실패: {response.status_code}")
        print(f"   응답: {response.text}")


def check_metadata_db_status(metadata_db: MetadataDBService):
    """메타DB 상태 확인"""
    print("=" * 80)
    print("📊 메타DB 상태")
    print("=" * 80)
    print()
    
    all_docs = metadata_db.get_all_documents()
    active_docs = [doc for doc in all_docs if doc.is_active]
    inactive_docs = [doc for doc in all_docs if not doc.is_active]
    
    print("📈 문서 통계:")
    print(f"  - 총 문서 수: {len(all_docs)}개")
    print(f"  - 활성 문서: {len(active_docs)}개")
    print(f"  - 비활성 문서: {len(inactive_docs)}개")
    print()
    
    if all_docs:
        print("📋 문서 목록:")
        print()
        
        # 활성 문서
        if active_docs:
            print("✅ 활성 문서:")
            for i, doc in enumerate(sorted(active_docs, key=lambda x: x.created_at), 1):
                print(f"  {i}. {doc.filename}")
                print(f"     - ID: {doc.id}")
                print(f"     - 회차: {doc.assembly_number}")
                print(f"     - 위원회: {doc.committee}")
                print(f"     - 회의번호: {doc.meeting_number}")
                print(f"     - 날짜: {doc.date}")
                print(f"     - File ID: {doc.vector_store_file_id}")
                print(f"     - 생성일: {doc.created_at}")
                print()
        
        # 비활성 문서
        if inactive_docs:
            print("⏸️ 비활성 문서:")
            for i, doc in enumerate(sorted(inactive_docs, key=lambda x: x.created_at), 1):
                print(f"  {i}. {doc.filename}")
                print(f"     - ID: {doc.id}")
                print(f"     - File ID: {doc.vector_store_file_id}")
                print()
    else:
        print("⚠️ 메타DB에 문서가 없습니다.")


def check_sync_status(vector_store: VectorStoreService, metadata_db: MetadataDBService):
    """Vector Store와 메타DB 동기화 상태 확인"""
    print("=" * 80)
    print("🔄 동기화 상태 확인")
    print("=" * 80)
    print()
    
    # 메타DB의 모든 파일 ID
    all_docs = metadata_db.get_all_documents()
    meta_file_ids = set([doc.vector_store_file_id for doc in all_docs if doc.vector_store_file_id])
    
    # Vector Store의 파일 ID
    if vector_store.vector_store_id:
        files_url = f"{vector_store.base_url}/vector_stores/{vector_store.vector_store_id}/files"
        files_response = requests.get(files_url, headers=vector_store.headers, params={"limit": 100})
        
        if files_response.status_code == 200:
            files_data = files_response.json()
            vs_file_ids = set([f["id"] for f in files_data.get("data", [])])
            
            print(f"📊 동기화 상태:")
            print(f"  - 메타DB 파일 수: {len(meta_file_ids)}개")
            print(f"  - Vector Store 파일 수: {len(vs_file_ids)}개")
            print()
            
            # 차이 확인
            only_in_meta = meta_file_ids - vs_file_ids
            only_in_vs = vs_file_ids - meta_file_ids
            
            if only_in_meta:
                print(f"⚠️ 메타DB에만 있는 파일 ({len(only_in_meta)}개):")
                for file_id in only_in_meta:
                    print(f"  - {file_id}")
                print()
            
            if only_in_vs:
                print(f"⚠️ Vector Store에만 있는 파일 ({len(only_in_vs)}개):")
                for file_id in only_in_vs:
                    print(f"  - {file_id}")
                print()
            
            if not only_in_meta and not only_in_vs:
                print("✅ Vector Store와 메타DB가 완전히 동기화되어 있습니다!")
        else:
            print(f"⚠️ Vector Store 파일 목록 조회 실패: {files_response.status_code}")


def main():
    """메인 함수"""
    print("=" * 80)
    print("Vector Store 및 메타DB 업로드 상태 확인")
    print("=" * 80)
    print(f"확인 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 서비스 초기화
    print("🔧 서비스 초기화 중...")
    metadata_db = MetadataDBService()
    vector_store = VectorStoreService(metadata_db=metadata_db)
    print("✅ 서비스 초기화 완료\n")
    
    # 1. Vector Store 상태 확인
    check_vector_store_status(vector_store)
    
    # 2. 메타DB 상태 확인
    check_metadata_db_status(metadata_db)
    
    # 3. 동기화 상태 확인
    check_sync_status(vector_store, metadata_db)
    
    print("=" * 80)
    print("✅ 확인 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

