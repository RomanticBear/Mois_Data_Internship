# Assembly Meeting RAG

> **최신 업데이트**: Vector Store 통합 완료! 🚀
> - 파일 업로드 시 자동 임베딩 및 Vector Store 저장
> - 질문 처리 속도 대폭 개선
> - FAISS 불필요 (OpenAI Vector Store 사용)

국회회의록 PDF 원문 기반 RAG 챗봇 - "최근 회의를 이해하고, 다음 회의를 준비해주는 AI"

## 프로젝트 개요

이 프로젝트는 OpenAI File Search 기반의 Managed RAG를 활용하여, 최신 국회회의록 PDF를 기반으로 다음 회의를 준비할 수 있는 특화 챗봇을 구현합니다.

### 핵심 목표

- ✅ LLM 파인튜닝 없이 **Managed RAG** 방식 활용
- ✅ 국회회의록이라는 특정 도메인에 최적화된 UX/워크플로우 제공
- ✅ "최근 회의 맥락을 이해하고 다음 회의를 준비해주는 AI"

### 핵심 활용 시나리오

- 최근 회의에서:
  - 어떤 쟁점이 나왔는지?
  - 누가 어떤 발언을 했는지?
  - 어떤 자료요구가 있었는지?

- 다음 회의에서:
  - 이어서 논의될 가능성이 있는 쟁점은 무엇인지?
  - 아직 해결되지 않은 이슈는 무엇인지?
  - 추가로 준비해야 할 자료는 무엇인지?

## 기술 스택

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, JavaScript, CSS
- **RAG**: OpenAI File Search (Managed RAG)
- **Vector Store**: OpenAI Vector Store
- **Database**: SQLite (메타DB)

## 프로젝트 구조

```
Assembly_Meeting_Rag/
├── backend/              # 백엔드 서버
│   ├── app/
│   │   ├── main.py      # FastAPI 메인 애플리케이션
│   │   ├── models/      # 데이터 모델
│   │   ├── services/    # 비즈니스 로직
│   │   │   ├── vector_store.py    # Vector Store 관리
│   │   │   ├── metadata_db.py     # 메타DB 관리
│   │   │   ├── active_window.py   # Active Window 관리
│   │   │   └── prompt_manager.py  # 프롬프트 관리
│   │   ├── api/         # API 엔드포인트
│   │   └── utils/       # 유틸리티 함수
│   ├── requirements.txt
│   └── .env.example
├── frontend/            # 프론트엔드 UI
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── templates/       # 질문 템플릿
├── data/                # 데모 데이터 (PDF 파일들)
├── tests/               # 테스트 코드
└── README.md
```

## 주요 기능

### 1. 문서 관리
- PDF 업로드 및 Vector Store 등록
- Active Window 관리 (슬라이딩 윈도우)
- 메타DB를 통한 문서 상태 관리

### 2. 질문 처리
- 질문 유형 자동 분류
- OpenAI File Search 기반 검색
- 도메인 특화 프롬프트 적용
- 구조화된 답변 생성

### 3. UI 기능
- 회차/위원회/날짜 선택
- 최신 회의 목록 표시
- 질문 입력 및 템플릿 제공
- 답변 결과 저장/공유

## 데이터 운영 전략

### 슬라이딩 윈도우 방식

- **Active Window**: 최근 N회차 회의록만 Vector Store에 유지
- **Cold Storage**: 그 이전 회의록은 메타DB에만 기록 (필요시 재업로드)

### 운영 정책

1. 새로운 회의 종료 시:
   - 최신 회의록 PDF 업로드
   - Active Window(Vector Store)에 추가
   - 가장 오래된 회의록 PDF 제거 (Active Window 유지)

2. 항상 "최근 회의 맥락만 유지"하여 비용 절감 및 검색 정확도 향상

## 설치 및 실행

### 1. 환경 설정

```bash
cd backend
cp .env.example .env
# .env 파일에 OpenAI API 키 설정
```

### 2. 의존성 설치

```bash
cd backend
pip install -r requirements.txt
```

### 3. 서버 실행

```bash
cd backend
uvicorn app.main:app --reload
```

### 4. 프론트엔드 접속

브라우저에서 `frontend/index.html` 파일 열기

## API 엔드포인트 (예정)

- `POST /api/upload` - PDF 업로드
- `POST /api/query` - 질문 처리
- `GET /api/meetings` - 회의 목록 조회
- `DELETE /api/meetings/{id}` - 회의록 삭제
- `GET /api/metadata` - 메타데이터 조회

## 향후 개발 계획

- [ ] 질문 템플릿 초안 작성
- [ ] Active Window 기준 결정 (최근 N회차)
- [ ] API 설계 완료
- [ ] PoC 서버 구축 및 실제 회의록 테스트
- [ ] 질문 유형별 프롬프트 최적화
- [ ] UI/UX 개선

## 라이선스

MIT





