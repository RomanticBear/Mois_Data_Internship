"""
PDF 업로드 API
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.models.document import DocumentUploadRequest, DocumentResponse
from app.services.active_window import ActiveWindowService
from app.services.vector_store import VectorStoreService
from app.services.metadata_db import MetadataDBService
from app.utils.filename_parser import create_metadata_from_filename
import os
import tempfile
from datetime import datetime

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


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    assembly_number: str = None,
    session_type: str = None,
    committee: str = None,
    meeting_number: int = None,
    date: str = None
):
    """
    PDF 파일 업로드 및 Vector Store 등록
    
    메타데이터는 파일명에서 추출하거나 요청 파라미터로 받을 수 있음
    """
    # 파일명 검증
    if not file.filename.endswith(('.pdf', '.PDF')):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # 서비스 가져오기
        metadata_db, vector_store, active_window = get_services()
        
        # 메타데이터 생성 (파일명에서 추출 또는 요청 파라미터 사용)
        parsed_metadata = create_metadata_from_filename(file.filename)
        
        # 요청 파라미터가 있으면 우선 사용
        metadata = {
            "filename": file.filename,
            "assembly_number": assembly_number or parsed_metadata["assembly_number"],
            "session_type": session_type or parsed_metadata["session_type"],
            "committee": committee or parsed_metadata["committee"],
            "meeting_number": meeting_number if meeting_number is not None else parsed_metadata["meeting_number"],
            "date": date or parsed_metadata["date"] or datetime.now().strftime("%Y.%m.%d"),
            "vector_store_file_id": None,
            "is_active": True
        }
        
        from app.models.document import DocumentMetadata
        doc_metadata = DocumentMetadata(**metadata)
        
        # Vector Store에 업로드
        file_id = vector_store.upload_file(tmp_file_path, doc_metadata)
        doc_metadata.vector_store_file_id = file_id
        
        # Active Window에 추가 (자동으로 크기 유지)
        created_doc = active_window.add_document(doc_metadata)
        
        return DocumentResponse(
            id=created_doc.id,
            filename=created_doc.filename,
            assembly_number=created_doc.assembly_number,
            session_type=created_doc.session_type,
            committee=created_doc.committee,
            meeting_number=created_doc.meeting_number,
            date=created_doc.date,
            is_active=created_doc.is_active,
            created_at=created_doc.created_at
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 실패: {str(e)}")
    
    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

