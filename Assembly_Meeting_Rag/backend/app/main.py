"""
Assembly Meeting RAG Backend
FastAPI 메인 애플리케이션
"""
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Assembly Meeting RAG API",
    description="국회회의록 PDF 원문 기반 RAG 챗봇 API",
    version="0.1.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경용, 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (프론트엔드)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def root():
    """루트 엔드포인트 - 프론트엔드로 리다이렉트"""
    from fastapi.responses import FileResponse
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "Assembly Meeting RAG API",
        "version": "0.1.0",
        "frontend": "Please access /static/index.html"
    }

@app.get("/style.css")
async def frontend_style():
    """프론트엔드 스타일 시트 제공 (정적 서버 없이 접근용)"""
    from fastapi.responses import FileResponse
    css_file = FRONTEND_DIR / "style.css"
    if css_file.exists():
        return FileResponse(str(css_file))
    return {"detail": "style.css not found"}

@app.get("/script.js")
async def frontend_script():
    """프론트엔드 스크립트 제공 (정적 서버 없이 접근용)"""
    from fastapi.responses import FileResponse
    js_file = FRONTEND_DIR / "script.js"
    if js_file.exists():
        return FileResponse(str(js_file))
    return {"detail": "script.js not found"}

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}

# API 라우터 추가
from app.api import upload, query, meetings, stats

app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(meetings.router, prefix="/api", tags=["meetings"])
app.include_router(stats.router, prefix="/api", tags=["stats"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

