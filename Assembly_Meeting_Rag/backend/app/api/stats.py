"""
통계 및 상태 정보 API
"""
from fastapi import APIRouter, HTTPException
from app.services.metadata_db import MetadataDBService
from app.services.vector_store import VectorStoreService
from typing import Dict, Any

router = APIRouter()

# 서비스 초기화 (지연 초기화)
_metadata_db = None
_vector_store = None

def get_services():
    """서비스 인스턴스 가져오기 (지연 초기화)"""
    global _metadata_db, _vector_store

    if _metadata_db is None:
        _metadata_db = MetadataDBService()
    if _vector_store is None:
        _vector_store = VectorStoreService(metadata_db=_metadata_db)

    return _metadata_db, _vector_store


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    시스템 통계 정보 조회
    """
    try:
        metadata_db, vector_store = get_services()

        all_docs = metadata_db.get_all_documents()

        committees = {}
        for doc in all_docs:
            if doc.committee not in committees:
                committees[doc.committee] = {"total": 0, "active": 0}
            committees[doc.committee]["total"] += 1
            committees[doc.committee]["active"] += 1

        assemblies = {}
        for doc in all_docs:
            if doc.assembly_number not in assemblies:
                assemblies[doc.assembly_number] = {"total": 0, "active": 0}
            assemblies[doc.assembly_number]["total"] += 1
            assemblies[doc.assembly_number]["active"] += 1

        return {
            "total_documents": len(all_docs),
            "active_documents": len(all_docs),
            "inactive_documents": 0,
            "window_size": None,
            "committees": committees,
            "assemblies": assemblies,
            "vector_store_id": vector_store.vector_store_id if vector_store.vector_store_id else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")
