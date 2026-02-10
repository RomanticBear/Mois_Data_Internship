"""
Vector Store 저장공간 및 OpenAI 사용량 확인
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "backend"))

from app.services.vector_store import VectorStoreService
from app.services.metadata_db import MetadataDBService

load_dotenv()

def check_vector_store_usage():
    """Vector Store 사용량 확인"""
    print("=" * 80)
    print("Vector Store 저장공간 및 OpenAI 사용량 확인")
    print("=" * 80)
    print()
    
    # 서비스 초기화
    metadata_db = MetadataDBService()
    vector_store = VectorStoreService(metadata_db=metadata_db)
    
    # Vector Store 정보 조회
    vs_url = f"{vector_store.base_url}/vector_stores/{vector_store.vector_store_id}"
    vs_response = requests.get(vs_url, headers=vector_store.headers)
    
    if vs_response.status_code == 200:
        vs_data = vs_response.json()
        file_counts = vs_data.get("file_counts", {})
        
        print("📊 Vector Store 정보:")
        print(f"   Vector Store ID: {vector_store.vector_store_id}")
        print(f"   이름: {vs_data.get('name', 'N/A')}")
        print(f"   생성일: {datetime.fromtimestamp(vs_data.get('created_at', 0))}")
        print()
        
        print("📈 파일 통계:")
        print(f"   - 총 파일 수: {file_counts.get('total', 0)}개")
        print(f"   - 완료: {file_counts.get('completed', 0)}개")
        print(f"   - 진행중: {file_counts.get('in_progress', 0)}개")
        print(f"   - 실패: {file_counts.get('failed', 0)}개")
        print()
        
        # Vector Store의 파일 목록 조회하여 용량 확인
        files_url = f"{vector_store.base_url}/vector_stores/{vector_store.vector_store_id}/files"
        files_response = requests.get(files_url, headers=vector_store.headers, params={"limit": 1000})
        
        total_bytes = 0
        if files_response.status_code == 200:
            files_data = files_response.json()
            files = files_data.get("data", [])
            
            print(f"📋 파일 상세 정보 ({len(files)}개):")
            for file_info in files[:10]:  # 처음 10개만 표시
                file_id = file_info.get("id", "N/A")
                status = file_info.get("status", "unknown")
                bytes_used = file_info.get("bytes", 0)
                total_bytes += bytes_used
                
                # 파일 정보 조회
                file_detail_url = f"{vector_store.base_url}/files/{file_id}"
                file_detail_response = requests.get(file_detail_url, headers=vector_store.headers)
                
                if file_detail_response.status_code == 200:
                    file_detail = file_detail_response.json()
                    filename = file_detail.get("filename", "N/A")
                    file_size = file_detail.get("bytes", 0)
                    print(f"   - {filename}: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            
            if len(files) > 10:
                print(f"   ... 외 {len(files) - 10}개 파일")
            print()
        
        # 전체 파일 크기 계산 (모든 파일 조회)
        print("📊 전체 파일 크기 계산 중...")
        all_files = []
        after = None
        while True:
            params = {"limit": 100}
            if after:
                params["after"] = after
            
            files_response = requests.get(files_url, headers=vector_store.headers, params=params)
            if files_response.status_code != 200:
                break
            
            files_data = files_response.json()
            batch_files = files_data.get("data", [])
            if not batch_files:
                break
            
            all_files.extend(batch_files)
            
            # 다음 페이지 확인
            has_more = files_data.get("has_more", False)
            if not has_more:
                break
            
            after = batch_files[-1].get("id")
        
        # 각 파일의 실제 크기 조회
        total_size_bytes = 0
        for file_info in all_files:
            file_id = file_info.get("id")
            file_detail_url = f"{vector_store.base_url}/files/{file_id}"
            file_detail_response = requests.get(file_detail_url, headers=vector_store.headers)
            
            if file_detail_response.status_code == 200:
                file_detail = file_detail_response.json()
                file_size = file_detail.get("bytes", 0)
                total_size_bytes += file_size
        
        print(f"✅ 총 파일 수: {len(all_files)}개")
        print(f"📦 총 저장공간: {total_size_bytes:,} bytes")
        print(f"   = {total_size_bytes / 1024 / 1024:.2f} MB")
        print(f"   = {total_size_bytes / 1024 / 1024 / 1024:.2f} GB")
        print()
    else:
        print(f"❌ Vector Store 정보 조회 실패: {vs_response.status_code}")
        print(f"   응답: {vs_response.text}")
        return
    
    # OpenAI 사용량 확인 (API를 통한 직접 확인은 제한적)
    print("=" * 80)
    print("📊 OpenAI 무료 플랜 제한사항")
    print("=" * 80)
    print()
    print("OpenAI 무료 플랜 (Tier 1) 제한:")
    print("  - 파일 업로드: 파일당 최대 512MB")
    print("  - Vector Store: 파일 수 제한 없음 (하지만 총 용량 제한 있음)")
    print("  - API 호출: 월별 제한 있음")
    print()
    print("⚠️  정확한 사용량 및 제한사항은 OpenAI 대시보드에서 확인하세요:")
    print("   https://platform.openai.com/usage")
    print()
    
    # 메타DB에서 등록된 파일 수 확인
    all_docs = metadata_db.get_all_documents()
    docs_with_file_id = [doc for doc in all_docs if doc.vector_store_file_id]
    
    print("📋 메타DB 통계:")
    print(f"   - 총 문서 수: {len(all_docs)}개")
    print(f"   - Vector Store에 등록된 문서: {len(docs_with_file_id)}개")
    print()
    
    print("=" * 80)
    print("✅ 확인 완료!")
    print("=" * 80)

if __name__ == "__main__":
    check_vector_store_usage()




