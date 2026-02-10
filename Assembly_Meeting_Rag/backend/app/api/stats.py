"""
통계 및 상태 정보 API
"""
from fastapi import APIRouter, HTTPException
from app.services.metadata_db import MetadataDBService
from app.services.vector_store import VectorStoreService
from app.services.active_window import ActiveWindowService
from typing import Dict, Any

router = APIRouter()

# 서비스 초기화 (지연 초기화)
_metadata_db = None
_vector_store = None
_active_window = None

def get_services():
    """서비스 인스턴스 가져오기 (지연 초기화)"""
    global _metadata_db, _vector_store, _active_window
    
    if _metadata_db is None:
        _metadata_db = MetadataDBService()
    if _vector_store is None:
        _vector_store = VectorStoreService(metadata_db=_metadata_db)
    if _active_window is None:
        _active_window = ActiveWindowService(metadata_db=_metadata_db, vector_store=_vector_store)
    
    return _metadata_db, _vector_store, _active_window


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    시스템 통계 정보 조회
    """
    try:
        metadata_db, vector_store, active_window = get_services()
        
        # 메타DB 통계
        all_docs = metadata_db.get_all_documents()
        active_docs = [doc for doc in all_docs if doc.is_active]
        inactive_docs = [doc for doc in all_docs if not doc.is_active]
        
        # 위원회별 통계
        committees = {}
        for doc in all_docs:
            if doc.committee not in committees:
                committees[doc.committee] = {"total": 0, "active": 0}
            committees[doc.committee]["total"] += 1
            if doc.is_active:
                committees[doc.committee]["active"] += 1
        
        # 회차별 통계
        assemblies = {}
        for doc in all_docs:
            if doc.assembly_number not in assemblies:
                assemblies[doc.assembly_number] = {"total": 0, "active": 0}
            assemblies[doc.assembly_number]["total"] += 1
            if doc.is_active:
                assemblies[doc.assembly_number]["active"] += 1
        
        # Active Window 정보
        window_size = active_window.get_window_size()
        
        return {
            "total_documents": len(all_docs),
            "active_documents": len(active_docs),
            "inactive_documents": len(inactive_docs),
            "window_size": window_size,
            "committees": committees,
            "assemblies": assemblies,
            "vector_store_id": vector_store.vector_store_id if vector_store.vector_store_id else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

