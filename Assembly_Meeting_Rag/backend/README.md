# Backend - Assembly Meeting RAG

FastAPI 기반 백엔드 서버

## 설치

```bash
pip install -r requirements.txt
```

## 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```
OPENAI_API_KEY=your_openai_api_key_here
HOST=0.0.0.0
PORT=8000
ACTIVE_WINDOW_SIZE=5
DATABASE_URL=sqlite:///./metadata.db
```

## 실행

```bash
# 개발 서버 실행
uvicorn app.main:app --reload

# 프로덕션 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 프로젝트 구조

```
backend/
├── app/
│   ├── main.py              # FastAPI 메인 애플리케이션
│   ├── models/              # 데이터 모델
│   │   └── document.py
│   ├── services/            # 비즈니스 로직
│   │   ├── vector_store.py    # Vector Store 관리
│   │   ├── metadata_db.py     # 메타DB 관리
│   │   ├── active_window.py   # Active Window 관리
│   │   └── prompt_manager.py  # 프롬프트 관리
│   └── api/                 # API 엔드포인트
│       ├── upload.py
│       ├── query.py
│       └── meetings.py
└── requirements.txt
```

## 주요 API 엔드포인트

### 문서 업로드
```
POST /api/upload
Content-Type: multipart/form-data

file: PDF 파일
assembly_number: 국회 회차 (선택)
session_type: 회기 유형 (선택)
committee: 위원회 (선택)
meeting_number: 회의 번호 (선택)
date: 날짜 (선택)
```

### 질문 처리
```
POST /api/query
Content-Type: application/json

{
  "question": "질문 내용",
  "question_type": "질문 유형 (선택)",
  "include_inactive": false
}
```

### 회의록 목록 조회
```
GET /api/meetings?is_active=true&committee=행정안전위원회&assembly_number=제415회
```

### 회의록 삭제
```
DELETE /api/meetings/{meeting_id}
```

## 개발 노트

### Vector Store 초기화

서버 시작 시 자동으로 Vector Store를 생성하거나 기존 Vector Store를 사용합니다.

### 메타DB 초기화

SQLite 데이터베이스는 자동으로 생성됩니다. (`metadata.db` 파일)

### Active Window 관리

새 문서를 추가하면 자동으로 Active Window 크기를 유지합니다. 가장 오래된 문서는 비활성화됩니다.





