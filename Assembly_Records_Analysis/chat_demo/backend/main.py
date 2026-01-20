"""
FastAPI 백엔드 서버
RAG QA 시스템을 위한 REST API 엔드포인트 제공
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 프로젝트 루트를 sys.path에 추가
current_dir = Path(__file__).resolve().parent
# parents[0] = chat_demo, parents[1] = Assembly_Records_Analysis
project_root = current_dir.parents[1]

# analysis_scripts 경로 추가 (parliament_analysis 모듈이 있는 곳)
analysis_scripts_path = project_root / "analysis_scripts"
if analysis_scripts_path.exists():
    sys.path.insert(0, str(analysis_scripts_path))
    print(f"✅ analysis_scripts 경로 추가: {analysis_scripts_path}")
else:
    print(f"❌ analysis_scripts 경로를 찾을 수 없습니다: {analysis_scripts_path}")

# .env 파일 로드
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from openai import OpenAI

from parliament_analysis.data.db_client import SupabaseDBClient
from parliament_analysis.data.embedding_client import EmbeddingClient
from parliament_analysis.rag.qa_system import RAGQASystem
from parliament_analysis.rag.retriever import RAGRetriever

# FastAPI 앱 생성
app = FastAPI(title="RAG QA System API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수로 QA 시스템 저장
qa_system: Optional[RAGQASystem] = None


def initialize_qa_system():
    """QA 시스템 초기화"""
    global qa_system
    
    try:
        # API 키 확인
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. "
                ".env 파일에 OPENAI_API_KEY를 추가해주세요."
            )
        
        # 1. OpenAI 클라이언트 생성
        llm_client = OpenAI(api_key=openai_api_key)
        embedding_client = EmbeddingClient(openai_client=llm_client)
        
        # 2. Supabase 클라이언트 생성
        try:
            db_client = SupabaseDBClient.from_env()
        except Exception as e:
            raise ValueError(
                f"Supabase 클라이언트 초기화 실패: {str(e)}\n"
                ".env 파일에 SUPABASE_URL과 SUPABASE_KEY를 확인해주세요."
            )
        
        # 3. Retriever 생성
        retriever = RAGRetriever(
            db_client=db_client,
            embedding_client=embedding_client,
            verbose=True,  # 디버깅을 위해 활성화
        )
        
        # 4. QA 시스템 생성
        qa_system = RAGQASystem(
            retriever=retriever,
            llm_client=llm_client,
        )
        
        print("✅ QA 시스템 초기화 완료")
        return qa_system
        
    except Exception as e:
        error_msg = f"QA 시스템 초기화 실패: {str(e)}"
        print(f"❌ {error_msg}")
        raise ValueError(error_msg)


# 앱 시작 시 QA 시스템 초기화
@app.on_event("startup")
async def startup_event():
    try:
        initialize_qa_system()
    except Exception as e:
        print(f"⚠️  시작 시 QA 시스템 초기화 실패: {e}")
        print("⚠️  첫 번째 요청 시 다시 시도됩니다.")


# 요청 모델
class QuestionRequest(BaseModel):
    question: str
    session_name: Optional[str] = None


class QuestionResponse(BaseModel):
    answer: str
    sources: list
    question_type: Optional[str] = None


# 엔드포인트
@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "qa_system_initialized": qa_system is not None
    }


@app.get("/sessions")
async def get_sessions():
    """사용 가능한 세션 목록 반환"""
    # 하드코딩된 세션 리스트 (실제로는 DB에서 조회해야 함)
    sessions = [
        "제415회",
        "제416회",
        "제417회",
        "제418회",
        "제419회",
        "제420회",
        "제421회",
        "제422회",
        "제423회",
        "제424회",
    ]
    return {"sessions": sessions}


@app.post("/qa", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """질문에 대한 답변 생성"""
    global qa_system
    
    # QA 시스템이 초기화되지 않았으면 시도
    if qa_system is None:
        try:
            initialize_qa_system()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"QA 시스템 초기화 실패: {str(e)}"
            )
    
    try:
        result = qa_system.ask_question(
            request.question,
            session_name=request.session_name,
            top_k=3,
            include_sources=True,
        )
        
        return QuestionResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            question_type=result.get("question_type"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"답변 생성 중 오류 발생: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

