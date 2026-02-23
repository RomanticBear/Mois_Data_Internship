"""
Vector Store의 파일 목록을 메타DB(Supabase/Postgres)에 복구
"""
import os
from dotenv import load_dotenv
import requests

from app.services.vector_store import VectorStoreService
from app.services.metadata_db import MetadataDBService
from app.utils.filename_parser import create_metadata_from_filename
from app.models.document import DocumentMetadata


load_dotenv()


def fetch_vector_store_file_ids(vector_store: VectorStoreService) -> list[str]:
    """Vector Store에 등록된 파일 ID 전체 조회"""
    if not vector_store.vector_store_id:
        return []

    files_url = f"{vector_store.base_url}/vector_stores/{vector_store.vector_store_id}/files"
    file_ids: list[str] = []
    after = None

    while True:
        params = {"limit": 100}
        if after:
            params["after"] = after

        response = requests.get(files_url, headers=vector_store.headers, params=params)
        if response.status_code != 200:
            print(f"❌ 파일 목록 조회 실패: {response.status_code} - {response.text}")
            break

        data = response.json()
        batch = data.get("data", [])
        if not batch:
            break

        file_ids.extend([item.get("id") for item in batch if item.get("id")])

        if not data.get("has_more"):
            break

        after = batch[-1].get("id")

    return file_ids


def fetch_filename(vector_store: VectorStoreService, file_id: str) -> str | None:
    """OpenAI Files API에서 파일명 조회"""
    detail_url = f"{vector_store.base_url}/files/{file_id}"
    response = requests.get(detail_url, headers=vector_store.headers)
    if response.status_code != 200:
        print(f"⚠️ 파일 상세 조회 실패: {file_id} ({response.status_code})")
        return None
    return response.json().get("filename")


def main():
    print("=" * 80)
    print("Vector Store → 메타DB 복구")
    print("=" * 80)

    metadata_db = MetadataDBService()
    vector_store = VectorStoreService(metadata_db=metadata_db)

    if not vector_store.vector_store_id:
        print("❌ Vector Store ID를 찾을 수 없습니다.")
        return

    existing_ids = set(metadata_db.get_all_file_ids())
    print(f"✅ 기존 메타DB 파일 수: {len(existing_ids)}개")

    file_ids = fetch_vector_store_file_ids(vector_store)
    print(f"✅ Vector Store 파일 수: {len(file_ids)}개")

    created = 0
    skipped = 0
    failed = 0

    for file_id in file_ids:
        if file_id in existing_ids:
            skipped += 1
            continue

        filename = fetch_filename(vector_store, file_id)
        if not filename:
            failed += 1
            continue

        metadata = create_metadata_from_filename(filename)
        metadata["vector_store_file_id"] = file_id

        try:
            doc = DocumentMetadata(**metadata)
            metadata_db.create_document(doc)
            created += 1
        except Exception as e:
            failed += 1
            print(f"⚠️ 메타DB 저장 실패: {filename} ({file_id}) - {e}")

    print()
    print(f"✅ 복구 완료: 생성 {created} / 건너뜀 {skipped} / 실패 {failed}")
    print("=" * 80)


if __name__ == "__main__":
    main()
