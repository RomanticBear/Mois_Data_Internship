"""
질문 처리 API
"""
from fastapi import APIRouter, HTTPException
from app.models.document import QueryRequest, QueryResponse
from app.services.vector_store import VectorStoreService
from app.services.prompt_manager import PromptManager, QuestionType
from app.services.metadata_db import MetadataDBService

router = APIRouter()

# 서비스 초기화 (지연 초기화)
_metadata_db = None
_vector_store = None
_prompt_manager = None

def get_services():
    """서비스 인스턴스 가져오기 (지연 초기화)"""
    global _metadata_db, _vector_store, _prompt_manager
    
    if _metadata_db is None:
        _metadata_db = MetadataDBService()
    if _vector_store is None:
        # 메타DB 참조를 전달하여 Active 파일 ID 동기화
        _vector_store = VectorStoreService(metadata_db=_metadata_db)
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    
    return _metadata_db, _vector_store, _prompt_manager


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    질문 처리 및 답변 생성
    """
    try:
        # 서비스 가져오기
        metadata_db, vector_store, prompt_manager = get_services()
        
        # 질문 유형 분류
        question_type = None
        if request.question_type:
            try:
                # 유효한 QuestionType인지 확인
                question_type = QuestionType(request.question_type)
            except (ValueError, KeyError):
                # 유효하지 않은 값이면 자동 분류 사용
                question_type = prompt_manager.classify_question(request.question)
        else:
            # question_type이 없으면 자동 분류
            question_type = prompt_manager.classify_question(request.question)
        
        # 프롬프트 생성
        prompt = prompt_manager.get_prompt(question_type, request.question)
        
        # Active Window 파일 ID 목록 가져오기
        active_file_ids = None
        if request.include_inactive:
            # 전체 파일 검색 (Active + Inactive)
            active_file_ids = metadata_db.get_all_file_ids()
        else:
            # Active Window 파일만 검색
            active_file_ids = metadata_db.get_active_file_ids()
        
        # Vector Store에서 질문 처리
        result = vector_store.query(
            question=prompt,
            active_files_only=not request.include_inactive,
            file_ids=active_file_ids
        )
        
        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            question_type=question_type.value,
            metadata={
                "thread_id": result.get("thread_id"),
                "run_id": result.get("run_id")
            }
        )
    
    except Exception as e:
        error_msg = str(e)
        # OpenAI API 할당량 초과 에러인 경우 더 명확한 메시지 제공
        if "quota" in error_msg.lower() or "rate_limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail=f"OpenAI API 할당량 초과: {error_msg}. OpenAI 대시보드에서 사용량을 확인하고 결제 정보를 업데이트해주세요."
            )
        raise HTTPException(status_code=500, detail=f"질문 처리 실패: {error_msg}")

