"""
회의록 관리 API
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.models.document import DocumentResponse
from app.services.metadata_db import MetadataDBService
from app.services.active_window import ActiveWindowService
from app.services.vector_store import VectorStoreService

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
        # 메타DB 참조를 전달하여 Active 파일 ID 동기화
        _vector_store = VectorStoreService(metadata_db=_metadata_db)
    if _active_window is None:
        _active_window = ActiveWindowService(metadata_db=_metadata_db, vector_store=_vector_store)
    
    return _metadata_db, _vector_store, _active_window


@router.get("/meetings", response_model=List[DocumentResponse])
async def get_meetings(
    is_active: Optional[bool] = Query(None, description="Active Window 포함 여부"),
    committee: Optional[str] = Query(None, description="위원회 필터"),
    assembly_number: Optional[str] = Query(None, description="국회 회차 필터")
):
    """
    회의록 목록 조회
    """
    try:
        metadata_db, _, _ = get_services()
        documents = metadata_db.get_all_documents(
            is_active=is_active,
            committee=committee,
            assembly_number=assembly_number
        )
        
        return [
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                assembly_number=doc.assembly_number,
                session_type=doc.session_type,
                committee=doc.committee,
                meeting_number=doc.meeting_number,
                date=doc.date,
                is_active=doc.is_active,
                created_at=doc.created_at
            )
            for doc in documents
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 실패: {str(e)}")


@router.get("/meetings/{meeting_id}", response_model=DocumentResponse)
async def get_meeting(meeting_id: int):
    """
    특정 회의록 조회
    """
    metadata_db, _, _ = get_services()
    doc = metadata_db.get_document(meeting_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="회의록을 찾을 수 없습니다.")
    
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        assembly_number=doc.assembly_number,
        session_type=doc.session_type,
        committee=doc.committee,
        meeting_number=doc.meeting_number,
        date=doc.date,
        is_active=doc.is_active,
        created_at=doc.created_at
    )


@router.delete("/meetings/{meeting_id}")
async def delete_meeting(meeting_id: int):
    """
    회의록 삭제 (메타DB 및 Vector Store에서 제거)
    """
    metadata_db, vector_store, _ = get_services()
    doc = metadata_db.get_document(meeting_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="회의록을 찾을 수 없습니다.")
    
    try:
        # Vector Store에서 파일 제거
        if doc.vector_store_file_id:
            vector_store.delete_file(doc.vector_store_file_id)
        
        # 메타DB에서 삭제
        metadata_db.delete_document(meeting_id)
        
        return {"message": "회의록이 삭제되었습니다."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"삭제 실패: {str(e)}")


@router.get("/meetings/active/window-size")
async def get_window_size():
    """
    Active Window 크기 조회
    """
    _, _, active_window = get_services()
    return {"window_size": active_window.get_window_size()}


@router.put("/meetings/active/window-size")
async def set_window_size(size: int = Query(..., ge=1, description="Active Window 크기")):
    """
    Active Window 크기 설정
    """
    try:
        _, _, active_window = get_services()
        active_window.set_window_size(size)
        return {"message": f"Active Window 크기가 {size}로 설정되었습니다.", "window_size": size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"설정 실패: {str(e)}")

