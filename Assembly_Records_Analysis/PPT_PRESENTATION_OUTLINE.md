# 국회 회의록 OpenAI 분석 및 슈퍼베이스 저장 과정
## PPT 발표 장표 구성안

---

## 1장: 프로젝트 개요 및 목적

### 제목: 국회 회의록 AI 분석 시스템

**목적:**
- OpenAI API를 활용한 국회 회의록 구조화 분석
- 정당별 입장, 이슈 추출, QA 효과성 평가
- 벡터 임베딩 기반 RAG(Retrieval-Augmented Generation) 시스템 구축
- Supabase를 통한 구조화된 데이터 저장 및 검색 가능한 형태로 저장

**주요 기능:**
- 세션 요약 및 핵심 이슈 추출
- 정당별 입장 분석 (비판적/지지적/중립적/건의적)
- 질의-응답 효과성 평가
- 벡터 임베딩을 통한 의미 기반 검색 지원

---

## 2장: 전체 프로세스 플로우

### 데이터 처리 파이프라인

**슬라이드 플로우 다이어그램 (5단계):**

```
[01 데이터 수집] → [02 프롬프트 분석] → [03 임베딩 생성] → [04 벡터 DB 적재] → [05 RAG 구축]
```

**각 단계별 상세 내용:**

**01. 데이터 수집 (코드 실행)**
- 원본 회의록 데이터 수집
- 데이터 검증 및 전처리
- 발언 데이터 구조화
  - `session_name` - 회차 정보
  - `party` - 정당 정보
  - `speaker_name` - 발언자 이름
  - `agenda_item_titles` - 안건명
  - `speech_text` - 발언 내용

**02. 프롬프트 분석 (API)**
- 품질 필터링: 의사진행 발언 제거, 최소 50자 이상 발언만 선별
- 중요도 기반 선별: 정책 키워드, 질문/답변 패턴, 숫자/통계 포함 여부로 중요도 점수 계산
- OpenAI API를 활용한 회의록 분석
  - 세션 요약 분석 - 핵심 이슈, 정당별 입장 (최대 50개 중요 발언)
  - 정당별 입장 분석 - 비판적/지지적/중립적/건의적 (정당당 최대 10개)
  - QA 효과성 분석 - 질의-응답 품질 평가
- JSON 형식으로 구조화된 결과 반환

**03. 임베딩 생성**
- OpenAI Embeddings API (text-embedding-3-small) 활용
- 분석 결과 텍스트를 벡터로 변환
  - 세션 요약 임베딩
  - 안건별 요약 임베딩
  - 정당별 입장 임베딩
  - QA 질문/답변 임베딩
- 1536차원 벡터 생성

**04. 벡터 DB 적재 (Supabase)**
- Supabase PostgreSQL + pgvector 확장 활용
- 구조화된 데이터 및 벡터 임베딩 저장
  - sessions 테이블 (세션 메타데이터 + 요약 임베딩)
  - agenda_items 테이블 (안건 정보 + 임베딩)
  - party_positions 테이블 (정당 입장 + 임베딩)
  - qa_interactions 테이블 (QA 상호작용 + 질문/답변 임베딩)
- 벡터 검색 인덱스 생성

**05. RAG 구축**
- RAG 문서 청킹 (800자 청크, 100자 오버랩)
- 세션 요약, 정당 입장, QA 페어를 청크 단위로 분할
- 각 청크 임베딩 생성 및 documents_rag 테이블 저장
- 의미 기반 검색 및 질의-응답 시스템 구축

---

### 상세 데이터 처리 파이프라인 (4단계)

```
[01. 데이터 수집 및 전처리]
    ├─ 원본 회의록 데이터 수집
    ├─ 데이터 검증 및 구조화
    └─ 발언 데이터 분류
        ├─ `session_name` - 회차 정보
        ├─ `party` - 정당 정보
        ├─ `speaker_name` - 발언자 이름
        ├─ `agenda_item_titles` - 안건명
        └─ `speech_text` - 발언 내용
    ↓
[02. OpenAI API 분석 및 파싱]
    ├─ 품질 필터링 (의사진행 발언 제거, 최소 50자 이상)
    ├─ 중요도 기반 선별 (정책 키워드, 질문/답변 패턴, 통계 포함 여부)
    ├─ 세션 요약 분석 (핵심 이슈, 정당별 입장) - 최대 50개 중요 발언
    ├─ 정당별 입장 분석 (비판적/지지적/중립적/건의적) - 정당당 최대 10개
    ├─ QA 효과성 분석 (질의-응답 품질 평가)
    └─ JSON 응답 파싱 및 데이터 클래스 변환
    ↓
[03. 임베딩 벡터화]
    ├─ 세션 요약 임베딩 생성
    ├─ 안건별 요약 임베딩 생성
    ├─ 정당 입장 임베딩 생성
    ├─ QA 질문/답변 임베딩 생성
    └─ RAG 문서 청킹 및 청크 임베딩 생성
    ↓
[04. Supabase 벡터 DB 저장]
    ├─ sessions 테이블 (세션 메타데이터 + 요약 임베딩)
    ├─ agenda_items 테이블 (안건 정보 + 임베딩)
    ├─ party_positions 테이블 (정당 입장 + 임베딩)
    ├─ qa_interactions 테이블 (QA 상호작용 + 질문/답변 임베딩)
    └─ documents_rag 테이블 (RAG 청크 + 임베딩)
```

---

## 3장: OpenAI 분석 목적 및 전략

### 3-1. 분석 목적

**1. 세션 요약 분석 (Session Summary Analysis)**
- 회의의 전반적인 특징 파악
- 핵심 이슈 3-7개 추출 (중요도 기반 선별된 발언 분석)
- 정당별 주요 관심사 및 입장 파악
- 주요 쟁점 및 협력/대립 관계 분석
- 주요 사건 및 국회 대응 추출

**2. 정당별 입장 분석 (Party Position Analysis)**
- 안건별 정당 입장 분류 (비판적/지지적/중립적/건의적)
- 정당별 주요 포인트, 우려사항, 제안사항 추출
- 합의점 및 대립점 식별
- 협력 수준 평가

**3. QA 효과성 분석 (QA Effectiveness Analysis)**
- 질의-응답 품질 평가 (고/중/저품질)
- 질문 유형 분류 (정책 질의, 사실 확인, 비판 질의, 제안 질의)
- 응답 품질 지표 (완성도, 구체성, 응답성)
- 개선 제안 도출

### 3-2. 분석 전략

**품질 필터링:**
- 의사진행 발언 패턴 제거 - 15개 이상 패턴 감지
- 최소 50자 이상 발언만 선별
- 반복 문자 패턴 제거

**중요도 기반 선별:**
- 정책 키워드 포함 여부 - 30개 이상 키워드
- 질문/답변 패턴 감지
- 숫자/통계 포함 여부
- 문장 복잡도 평가
- 적절한 길이 평가 - 100-1000자 최적
- 중요도 점수 0.3 이상 발언만 선별

**질의-응답 형태 회의 특성 반영:**
- "반대/지지"가 아닌 "비판적/지지적/중립적/건의적" 관점
- 문제 제기, 건의, 평가 중심으로 분석

---

#*2 4장: OpenAI 프롬프트 상세

#  4-1. 세션 요약 분석 프롬프트

**입력 데이터:**
- 회차 정보, 총 발언 수
- 안건별 발언
- 정당별 발언

**데이터 출처:**
- 원본 데이터프레임: `session_name`, `agenda_item_titles`, `party`, `speaker_name`, `speech_text`
- 안건별 발언 통계: `agenda_item_titles` 집계
- 정당별 발언 통계: `party` 집계
- 정당별 중요 발언: `speech_text`에서 중요도 기반 선별 (최대 50개)

**처리 과정:**
- 품질 필터링: 의사진행 발언 제거, 최소 50자 이상
- 중요도 점수 계산: 정책 키워드, 질문/답변 패턴, 숫자/통계 포함, 문장 복잡도
- 선별 방식: 중요도 순 정렬, 정당별 균형 유지
- 발언 길이: 최대 500자

**프롬프트 구조:**
```
당신은 국회 회의록 분석 전문가입니다. 다음은 {session_name}의 회의록 데이터입니다.

=== 회차 정보 ===
회차: {session_name}
총 발언 수: {total_speeches}개

=== 안건 통계 ===
{agenda_stats}

=== 정당별 발언 통계 ===
{party_stats}

=== 대표 발언 샘플 ===
{speeches_sample}

=== 분석 요청 ===
다음 발언들을 분석하되, **의사진행 발언은 제외**하고 **정책 관련 실질적인 발언만** 포함하여 분석하세요.

다음 형식의 JSON으로 분석 결과를 제공해주세요:
{
  "session_name": "{session_name}",
  "key_issues": [
    {
      "issue": "이슈명",
      "importance": "높음/중간/낮음",
      "description": "이슈에 대한 설명",
      "mentioned_parties": ["정당1", "정당2"]
    }
  ],
  "party_positions": {
    "정당명": {
      "main_concerns": ["관심사1", "관심사2"],
      "key_statements": "주요 발언 요약",
      "stance": "비판적/지지적/중립적/건의적"
    }
  },
  "major_conflicts": [
    {
      "topic": "쟁점명",
      "parties_involved": ["정당1", "정당2"],
      "nature": "비판/협력/토론/질의"
    }
  ],
  "key_events": [
    {
      "event": "사건/참사명",
      "description": "설명",
      "response": "국회 대응"
    }
  ],
  "session_characteristics": "회차의 전반적인 특징 요약"
}
```

**시스템 프롬프트:**
- "당신은 국회 회의록 분석 전문가입니다. 정확하고 구조화된 JSON 형식으로 분석 결과를 제공합니다."

**API 설정:**
- Model: GPT-4 또는 GPT-3.5-turbo
- Temperature: 0.3 (일관성 있는 분석)
- Response Format: JSON Object

---

 4-2. 정당별 입장 분석 프롬프트

**입력 데이터:**
- 안건명, 안건별 총 발언 수
- 정당별 발언

**데이터 출처:**
- 원본 데이터프레임: `agenda_item_titles`, `party`, `speech_text`
- 안건별 필터링: `agenda_item_titles == agenda_title` 조건
- 안건별 총 발언 수: 필터링 후 집계
- 정당별 중요 발언: `speech_text`에서 중요도 기반 선별 (정당당 최대 10개)

**처리 과정:**
- 안건별 필터링 후, 정당별 중요도 순 정렬
- 품질 필터링: 의사진행 발언 제거, 최소 50자 이상
- 중요도 점수 계산: 정책 키워드, 질문/답변 패턴, 숫자/통계 포함, 문장 복잡도
- 발언 길이: 최대 500자

**프롬프트 구조:**
```
당신은 국회 회의록 분석 전문가입니다. 다음은 {session_name}의 안건 "{agenda_title}"에 대한 정당별 발언입니다.

=== 안건 정보 ===
안건명: {agenda_title}
총 발언 수: {total_speeches}개

=== 정당별 발언 샘플 ===
[정당: {party}]
1. {speech1}
2. {speech2}
...

=== 분석 요청 ===
이 회의는 입법 표결이 아니라 질의-응답 형태의 위원회 회의입니다.
다음 JSON 형식으로 정당별 관점을 분석해주세요:

{
  "agenda": "{agenda_title}",
  "party_positions": {
    "정당명": {
      "stance": "비판적/지지적/중립적/건의적",
      "key_points": ["주요 포인트1", "주요 포인트2"],
      "concerns": ["우려사항1", "우려사항2"],
      "suggestions": ["제안사항1", "제안사항2"],
      "key_statements": "주요 발언 요약"
    }
  },
  "consensus_points": ["합의점1", "합의점2"],
  "conflict_points": ["대립점1", "대립점2"],
  "cooperation_level": "높음/중간/낮음",
  "summary": "안건에 대한 종합 분석"
}
```

---
 4-3. QA 효과성 분석 프롬프트

**입력 데이터:**
- 질문자/답변자 정보 (정당, 발언자명)
- 질문 및 답변 텍스트

**데이터 출처:**
- 원본 데이터프레임: `speech_text`, `speaker_name`, `party`, `session_name`
- QA 페어 추출: 연속된 발언에서 질문/답변 마커로 페어 추출 (최대 10개)
  - 질문: 현재 발언의 `speech_text`, `speaker_name`, `party`
  - 답변: 다음 발언의 `speech_text`, `speaker_name`, `party`

**처리 과정:**
- QA 페어 추출: 연속된 발언에서 질문/답변 마커로 페어 추출
- 질문 마커: "질의", "질문", "?", "문의", "묻고 싶", "알고 싶"
- 답변 마커: "답변", "설명", "말씀", "드리", "알려"
- 발언 길이: 최대 500자

**프롬프트 구조:**
```
당신은 국회 회의록 분석 전문가입니다. 다음은 질의-응답 샘플입니다.

=== 질의-응답 샘플 ===
[질의-응답 1]
질문자 ({question_party}): {questioner}
질문: {question}
답변자 ({answer_party}): {answerer}
답변: {answer}
...

=== 분석 요청 ===
다음 JSON 형식으로 질의-응답 효과성을 분석해주세요:

{
  "session_name": "{session_name}",
  "total_qa_pairs": {total_count},
  "quality_distribution": {
    "high": "고품질 응답 비율 (%)",
    "medium": "중품질 응답 비율 (%)",
    "low": "저품질 응답 비율 (%)"
  },
  "question_types": {
    "policy_inquiry": "정책 질의 비율 (%)",
    "fact_checking": "사실 확인 비율 (%)",
    "criticism": "비판 질의 비율 (%)",
    "suggestion": "제안 질의 비율 (%)"
  },
  "answer_quality": {
    "completeness": "완성도 평균 (1-10)",
    "specificity": "구체성 평균 (1-10)",
    "responsiveness": "응답성 평균 (1-10)"
  },
  "key_issues": [
    {
      "issue": "주요 이슈",
      "qa_count": "질의-응답 수",
      "quality": "평균 품질"
    }
  ],
  "improvement_suggestions": ["개선 제안1", "개선 제안2"]
}
```

---

## 5장: 데이터 파싱 및 구조화

### 5-1. 분석 결과 파싱

**세션 요약 파싱:**
```python
SessionSummary(
    session_name: str
    meeting_date: datetime | None
    key_issues: List[Dict]  # 이슈명, 중요도, 설명, 언급 정당
    overall_sentiment: float | None
    raw_summary: str | None  # session_characteristics
    metadata: Dict  # party_positions, major_conflicts, key_events
)
```

**정당별 입장 파싱:**
```python
AgendaPartyAnalysis(
    session_name: str
    agenda_title: str
    party_positions: List[PartyPosition]
    consensus_points: List[str]
    conflict_points: List[str]
    cooperation_level: str | None
    summary_text: str | None
)

PartyPosition(
    session_name: str
    agenda_title: str
    party_name: str
    stance_label: str  # 비판적/지지적/중립적/건의적
    key_points: List[str]
    concerns: List[str]
    suggestions: List[str]
    summary_text: str | None
)
```

**QA 효과성 파싱:**
```python
QAAnalysisMetrics(
    session_name: str
    total_qa_pairs: int
    quality_distribution: Dict  # high, medium, low
    question_types: Dict  # policy_inquiry, fact_checking, etc.
    answer_quality: Dict  # completeness, specificity, responsiveness
    key_issues: List[Dict]
    improvement_suggestions: List[str]
)
```

### 5-2. 데이터 변환 과정

1. **JSON 응답 파싱:** OpenAI API JSON 응답을 Python 딕셔너리로 변환
2. **데이터 클래스 변환:** 딕셔너리를 dataclass 객체로 변환
3. **타입 검증:** meeting_date를 datetime으로 변환, 리스트 타입 검증
4. **메타데이터 보존:** raw_llm_response를 metadata에 저장

---

## 6장: 임베딩 생성 과정

### 6-1. 임베딩 개념 소개

**임베딩이란?**
- 텍스트를 수치 벡터로 변환하는 과정
- 의미적으로 유사한 텍스트는 벡터 공간에서 가까운 위치에 배치
- 벡터 검색을 통해 유사한 내용을 빠르게 찾을 수 있음

**임베딩 생성 도구:**
- **API**: OpenAI Embeddings API
- **모델**: `text-embedding-3-small`
- **벡터 차원**: 1536차원
- **특징**: 배치 처리 지원, 비용 효율적

**임베딩 생성 방법:**
```python
# OpenAI Embeddings API 호출
response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=text  # 또는 texts (배치 처리)
)
embedding = response.data[0].embedding  # 1536차원 벡터
```

**임베딩 생성 과정:**
```
텍스트 입력 → OpenAI Embeddings API 호출 → 1536차원 벡터 출력
```

**예시:**
```
입력: "화성 공장 화재와 관련된 정부의 대응 문제"
출력: [0.123, -0.456, 0.789, ..., 0.234] (1536개 숫자)
```

---

### 6-2. 임베딩 생성 대상 및 흐름

**임베딩 생성 대상 (5가지):**

**1. 세션, 안건별 요약**
- 전체 회의록의 핵심 이슈·쟁점·정당별 입장 및 각 안건의 합의점·대립점을 포함한 요약 텍스트로, 유사 회의록/안건 검색 및 주제별 분류에 활용

**2. 정당별 입장 요약**
- 각 정당의 주요 포인트·우려사항·제안사항·입장 레이블을 안건별로 분리하여 임베딩하며, 유사 정당 입장 검색 및 비교 분석에 활용

**3. QA 질문/답변**
- 정책 질의·사실 확인·비판 질의 등 질문과 정부/관계자 응답을 각각 임베딩하여, 유사 질문/답변 검색 및 QA 효과성 분석에 활용

**4. RAG 문서 청크**
- 세션 요약·정당 입장·QA 페어를 800자 단위로 청킹하고 100자 오버랩으로 문맥을 유지한 검색용 문서로, 사용자 질문에 대한 유사 문서 검색 후 답변 생성(RAG)에 활용

**임베딩 생성 흐름:**
```
[OpenAI API 분석 결과]
    ↓
[텍스트 추출]
    ↓
[임베딩 생성] ← OpenAI Embeddings API
    ↓
[벡터 저장] ← Supabase
```

---

### 6-3. A~D 직접 임베딩 생성 과정

### 6-2. 직접 임베딩 생성 (A~D)

**A. 세션 요약 임베딩**

**생성 시점:** OpenAI API 호출 후 JSON 파싱 직후

**텍스트 추출:**
```python
# raw_summary 우선 사용, 없으면 key_issues 조합
summary_text = session_summary.raw_summary
# 또는
summary_text = "\n".join(
    f"{issue.issue} - {issue.description}"
    for issue in session_summary.key_issues
)
```

**임베딩 생성:**
```python
summary_embedding = embedding_client.embed_text(summary_text)
```

**저장 위치:** `sessions.summary_embedding`

**목적:** 세션 요약 벡터 검색 (유사 세션 찾기)

---

**B. 안건별 요약 임베딩**

**생성 시점:** OpenAI API 호출 후 JSON 파싱 직후

**텍스트 추출:**
```python
# summary_text 우선 사용, 없으면 consensus_points + conflict_points 조합
agenda_summary = analysis.summary_text
# 또는
agenda_summary = "\n".join(
    analysis.consensus_points + analysis.conflict_points
)
```

**임베딩 생성:**
```python
agenda_embedding = embedding_client.embed_text(agenda_summary)
```

**저장 위치:** `agenda_items.summary_embedding`

**목적:** 안건 요약 벡터 검색 (유사 안건 찾기)

---

**C. 정당별 입장 임베딩**

**생성 시점:** OpenAI API 호출 후 JSON 파싱 직후

**텍스트 추출:**
```python
# summary_text 우선 사용, 없으면 key_points + concerns + suggestions 조합
summary_text = position.summary_text
# 또는
summary_text = "\n".join([
    position.key_points,
    position.concerns,
    position.suggestions
])
```

**임베딩 생성:**
```python
stance_embedding = embedding_client.embed_text(summary_text)
```

**저장 위치:** `party_positions.stance_embedding`

**목적:** 정당 입장 벡터 검색 (유사 입장 찾기)

---

**D. QA 질문/답변 임베딩**

**생성 시점:** OpenAI API 호출 후 JSON 파싱 직후

**텍스트 추출:**
```python
# 질문과 답변을 각각 추출
questions = [pair.get("question") for pair in qa_pairs]
answers = [pair.get("answer") for pair in qa_pairs]
```

**임베딩 생성 (배치 처리):**
```python
question_embeddings = embedding_client.embed_texts(questions)
answer_embeddings = embedding_client.embed_texts(answers)
```

**저장 위치:** `qa_interactions.question_embedding`, `qa_interactions.answer_embedding`

**목적:** QA 질문/답변 벡터 검색 (유사 질문/답변 찾기)

---

### 6-3. RAG 문서 청킹 및 임베딩 (E)

**생성 시점:** A~D 데이터가 Supabase에 저장된 후

**데이터 소스:** A~D의 분석 결과를 재사용하여 RAG 검색용 문서 생성

**청킹 처리 방식:**
- **모든 RAG 문서는 동일하게 청킹(길이, 오버랩)으로 처리됨**
- 청크 크기: 800자
- 오버랩: 100자
- 텍스트가 800자 이하면 그대로, 800자 초과면 800자 단위로 분할
- QA 페어도 질문+답변을 하나의 텍스트로 합친 후 청킹 처리 (페어 단위가 아닌 텍스트 길이 기준)

**E-1. 세션 요약 RAG 문서**

**데이터 소스:** A (세션 요약 분석 결과)

**텍스트 구성:**
```python
# 하나의 session_summary 객체의 여러 필드를 합쳐서 하나의 텍스트로 생성
summary_parts = [
    raw_summary,           # 세션 요약
    "핵심 이슈 요약:",
    *[각 key_issue],       # 핵심 이슈 목록
    "주요 쟁점:",
    *[각 major_conflict],  # 주요 쟁점
    "주요 사건:",
    *[각 key_event]        # 주요 사건
]
text = "\n".join(summary_parts)  # 모든 필드를 하나의 텍스트로 합침
```

**청킹 처리:**
- **하나의 세션 요약 객체의 모든 필드를 합쳐서 하나의 텍스트로 만든 후 청킹**
- 청크 크기: 800자
- 오버랩: 100자
- 텍스트가 800자 이하면 그대로, 800자 초과면 800자 단위로 분할

**임베딩 생성:**
```python
# 각 청크에 대해 임베딩 생성
for chunk in chunks:
    embedding = embedding_client.embed_text(chunk.content)
```

**저장 위치:** `documents_rag` 테이블
- `source_type`: "session_summary"
- `source_id`: "session::제415회"
- `chunk_index`: 0, 1, 2, ...

**목적:** RAG 검색을 위한 세션 요약 벡터 저장

---

**E-2. 정당별 입장 RAG 문서**

**데이터 소스:** C (정당별 입장 분석 결과)

**텍스트 구성:**
```python
# 여러 position(정당별 입장)이 있으면 각각 개별 처리
for position in positions:  # 각 정당별 입장마다 반복
    # 하나의 position의 여러 필드를 합쳐서 하나의 텍스트로 생성
    text_sections = [
        f"[안건] {agenda_title}",
        f"[정당] {party_name}",
        f"[입장] {stance_label}",
        summary_text,
        "주요 포인트:\n" + "\n".join(key_points),
        "우려 사항:\n" + "\n".join(concerns),
        "제안 사항:\n" + "\n".join(suggestions)
    ]
    text = "\n".join(text_sections)  # 하나의 position의 모든 필드를 합침
    # 각 position마다 개별적으로 청킹 처리
```

**청킹 처리:**
- **각 position(정당별 입장)마다 개별적으로 처리**
- 하나의 position의 모든 필드를 합쳐서 하나의 텍스트로 만든 후 청킹
- 여러 position이 있으면 각각 개별적으로 청킹 (모든 position을 합치지 않음)
- 청크 크기: 800자
- 오버랩: 100자
- 각 position의 텍스트가 800자 이하면 그대로, 800자 초과면 800자 단위로 분할

**임베딩 생성:**
```python
# 각 청크에 대해 임베딩 생성
for chunk in chunks:
    embedding = embedding_client.embed_text(chunk.content)
```

**저장 위치:** `documents_rag` 테이블
- `source_type`: "party_position"
- `source_id`: "session::제415회::agenda::안건명::party::정당명"
- `chunk_index`: 0, 1, 2, ...

**목적:** RAG 검색을 위한 정당 입장 벡터 저장

---

**E-3. QA 페어 RAG 문서**

**데이터 소스:** D (QA 페어 분석 결과)

**텍스트 구성:**
```python
# qa_pairs에서 추출하여 하나의 텍스트로 합침
text = (
    f"[질문자] {questioner} ({question_party})\n"
    f"[질문]\n{question}\n\n"
    f"[답변자] {answerer} ({answer_party})\n"
    f"[답변]\n{answer}"
)
# 질문과 답변을 하나의 텍스트로 합친 후 청킹 처리
```

**청킹 처리:**
- **각 QA 페어마다 개별적으로 처리**
- 하나의 QA 페어의 질문+답변을 합쳐서 하나의 텍스트로 만든 후 청킹
- 여러 QA 페어가 있으면 각각 개별적으로 청킹 (모든 페어를 합치지 않음)
- 청크 크기: 800자
- 오버랩: 100자
- 각 QA 페어의 텍스트가 800자 이하면 그대로, 800자 초과면 800자 단위로 분할

**임베딩 생성:**
```python
# 각 청크에 대해 임베딩 생성
for chunk in chunks:
    embedding = embedding_client.embed_text(chunk.content)
```

**저장 위치:** `documents_rag` 테이블
- `source_type`: "qa_pair"
- `source_id`: "session::제415회::qa::0"
- `chunk_index`: 0, 1, 2, ...

**목적:** RAG 검색을 위한 QA 페어 벡터 저장

---

### 6-4. A~D vs E 비교

| 항목 | A~D (직접 임베딩) | E (RAG 문서) |
|------|------------------|--------------|
| **생성 시점** | OpenAI API 호출 후 즉시 | A~D 저장 후 |
| **데이터 소스** | OpenAI API JSON 응답 | A~D 분석 결과 재사용 |
| **임베딩 대상** | 요약 텍스트 전체 | 청킹된 텍스트 (800자 단위) |
| **저장 위치** | 각 테이블 임베딩 컬럼 | `documents_rag` 테이블 |
| **목적** | 구조화된 데이터 저장 및 검색 | RAG 검색을 위한 벡터 저장 |
| **용도** | 유사 세션/안건/입장/QA 찾기 | 사용자 질문에 대한 유사 문서 검색 후 답변 생성 |

**왜 A~D와 E를 분리하는가?**
1. **A~D**: 구조화된 데이터 저장용 (각 테이블에 직접 저장, 빠른 검색)
2. **E**: RAG 검색용 (사용자 질문에 대한 유사 문서 검색 후 답변 생성)
3. **청킹**: 긴 텍스트를 작은 단위로 분할하여 정확한 벡터 검색 지원
4. **메타데이터**: RAG 문서에 source_type, source_id, chunk_index 등 상세 메타데이터 저장

---

### 6-5. 임베딩 생성 기술

**코드 위치:**
- **파일**: `parliament_analysis/data/embedding_client.py`
- **클래스**: `EmbeddingClient`
- **호출 위치**: `parliament_analysis/pipeline/persistence.py`의 `persist_analysis_to_supabase()` 함수

**임베딩 생성 도구:**
- **OpenAI Embeddings API**를 사용하여 텍스트를 벡터로 변환
- `EmbeddingClient` 클래스를 통해 OpenAI API를 래핑하여 사용

**벡터 차원:**
- **1536차원** (모델이 자동으로 결정, 우리가 지정하지 않음)
- `text-embedding-3-small` 모델의 고정된 출력 차원
- 모델 변경 시에만 차원이 변경됨

**임베딩 생성 시점 및 데이터:**

**시점**: `persist_analysis_to_supabase()` 함수 호출 시
- OpenAI API 분석 완료 후
- JSON 파싱 및 데이터 클래스 변환 후
- Supabase 저장 직전

**데이터 흐름:**
```
[OpenAI API 분석 결과]
    ↓
[Python 객체: SessionSummary, AgendaPartyAnalysis, qa_pairs]
    ↓
[텍스트 추출]
    ├─ A. 세션 요약: session_summary.raw_summary 또는 key_issues
    ├─ B. 안건 요약: analysis.summary_text 또는 consensus_points + conflict_points
    ├─ C. 정당 입장: position.summary_text 또는 key_points + concerns + suggestions
    └─ D. QA 페어: qa_pairs의 question, answer
    ↓
[임베딩 생성] ← EmbeddingClient.embed_text() 또는 embed_texts()
    ↓
[1536차원 벡터]
    ↓
[Supabase 저장]
```

**단일 텍스트 임베딩:**
```python
# 파일: parliament_analysis/data/embedding_client.py
def embed_text(self, text: str) -> List[float]:
    response = self.openai_client.embeddings.create(
        model="text-embedding-3-small",  # 모델 지정
        input=text,  # 텍스트 입력
    )
    return response.data[0].embedding  # 1536차원 벡터 반환 (자동)
```

**배치 텍스트 임베딩:**
```python
# 파일: parliament_analysis/data/embedding_client.py
def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
    response = self.openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=list(texts),  # 텍스트 리스트 입력
    )
    return [item.embedding for item in response.data]  # 각각 1536차원 벡터
```

**실제 호출 예시 (persistence.py):**
```python
# A. 세션 요약 임베딩 (114-116줄)
summary_embedding = embedding_client.embed_text(summary_text)

# B. 안건 요약 임베딩 (168-170줄)
agenda_embedding = embedding_client.embed_text(agenda_summary)

# C. 정당 입장 임베딩 (party_position_to_row 함수 내)
stance_embedding = embedding_client.embed_text(summary_text)

# D. QA 질문/답변 임베딩 (205-210줄, 배치 처리)
question_embeddings = embedding_client.embed_texts([pair.get("question") for pair in qa_pairs])
answer_embeddings = embedding_client.embed_texts([pair.get("answer") for pair in qa_pairs])

# E. RAG 문서 임베딩 (vector_store.upsert_documents 내부)
embeddings = embedding_client.embed_texts([item.content for item in items])
```

---

## 7장: Supabase 데이터베이스 스키마

### 7-1. sessions 테이블

**목적:** 세션 레벨 메타데이터 및 요약 정보 저장

**필드:**
| 필드명 | 타입 | 설명 |
|--------|------|------|
| session_id | uuid (PK) | 기본 키 |
| session_name | text (UNIQUE) | 회차 식별자 (예: "제415회") |
| meeting_date | date | 회의 일자 |
| source_path | text | 원본 데이터 경로 |
| hash_digest | text | 콘텐츠 변경 감지용 체크섬 |
| analysis_version | text | 분석 프롬프트 버전 |
| analyzed_at | timestamptz | 분석 실행 시간 |
| summary_text | text | 세션 요약 텍스트 |
| summary_embedding | vector(1536) | 세션 요약 임베딩 |
| metadata | jsonb | 추가 정보 (total_speeches, session_summary, qa_metrics 등) |

**인덱스:**
- session_name (UNIQUE)
- summary_embedding (벡터 검색용)

---

### 7-2. agenda_items 테이블

**목적:** 안건별 요약 및 임베딩 저장

**필드:**
| 필드명 | 타입 | 설명 |
|--------|------|------|
| agenda_id | uuid (PK) | 기본 키 |
| session_id | uuid (FK) | sessions 테이블 참조 |
| agenda_title | text | 안건명 |
| agenda_category | text | 안건 카테고리 |
| summary_text | text | 안건 요약 텍스트 |
| summary_embedding | vector(1536) | 안건 요약 임베딩 |
| metadata | jsonb | 추가 정보 (consensus_points, conflict_points, cooperation_level 등) |

**인덱스:**
- session_id
- agenda_title
- summary_embedding (벡터 검색용)

---

### 7-3. party_positions 테이블

**목적:** 안건별 정당 입장 정보 저장

**필드:**
| 필드명 | 타입 | 설명 |
|--------|------|------|
| agenda_id | uuid (FK, PK) | agenda_items 테이블 참조 (복합 키) |
| party_name | text (PK) | 정당명 (복합 키) |
| stance_label | text | 입장 레이블 (비판적/지지적/중립적/건의적) |
| key_points | jsonb | 주요 포인트 리스트 |
| concerns | jsonb | 우려사항 리스트 |
| suggestions | jsonb | 제안사항 리스트 |
| summary_text | text | 정당 입장 요약 텍스트 |
| stance_embedding | vector(1536) | 정당 입장 임베딩 |
| metadata | jsonb | 추가 정보 (speaker counts 등) |

**인덱스:**
- (agenda_id, party_name) (복합 기본 키)
- stance_embedding (벡터 검색용)

---

### 7-4. qa_interactions 테이블

**목적:** 질의-응답 상호작용 및 효과성 지표 저장

**필드:**
| 필드명 | 타입 | 설명 |
|--------|------|------|
| qa_id | uuid (PK) | 기본 키 |
| agenda_id | uuid (FK) | agenda_items 테이블 참조 |
| questioner | text | 질문자 이름 |
| respondent | text | 답변자 이름 |
| question_text | text | 질문 텍스트 |
| answer_text | text | 답변 텍스트 |
| effectiveness_score | numeric | 효과성 점수 (0-1 또는 0-100) |
| effectiveness_bucket | text | 효과성 카테고리 (high/medium/low) |
| tags | jsonb | 태그 리스트 (주제, 스타일 등) |
| question_embedding | vector(1536) | 질문 임베딩 |
| answer_embedding | vector(1536) | 답변 임베딩 |
| metadata | jsonb | 추가 정보 (question_party, answer_party, session_name 등) |

**인덱스:**
- agenda_id
- questioner
- respondent
- question_embedding (벡터 검색용)
- answer_embedding (벡터 검색용)

---

### 7-5. documents_rag 테이블

**목적:** RAG 검색을 위한 문서 청크 저장

**필드:**
| 필드명 | 타입 | 설명 |
|--------|------|------|
| document_id | uuid (PK) | 기본 키 |
| source_type | text | 소스 타입 (session_summary / party_position / qa_pair / transcript) |
| source_id | text | 원본 소스 식별자 (예: "session::제415회::agenda::안건명::party::정당명") |
| chunk_index | integer | 소스 문서 내 청크 인덱스 |
| content | text | 청크 텍스트 |
| metadata | jsonb | 메타데이터 (session_name, agenda_title, party_name, stance_label 등) |
| embedding | vector(1536) | 청크 임베딩 |

**인덱스:**
- (source_type, source_id)
- embedding (벡터 검색용)

**청킹 전략:**
- 청크 크기: 800자
- 오버랩: 100자
- 의미 단위 유지를 위한 텍스트 분할

---

## 8장: 데이터 저장 과정

### 8-1. 저장 프로세스 개요

**저장 순서:**
1. A~D 직접 임베딩 저장 (구조화된 데이터)
2. E RAG 문서 청킹 및 임베딩 저장 (검색용 벡터)

---

### 8-2. A~D 직접 임베딩 저장

**단계 1. 세션 레코드 저장 (A)**

**저장 시점:** OpenAI API 호출 후 JSON 파싱 직후

**저장 데이터:**
```python
session_record = {
    "session_name": session_name,
    "hash_digest": hash_digest,
    "analysis_version": analysis_version,
    "meeting_date": meeting_date,
    "summary_text": summary_text,  # raw_summary 또는 key_issues 조합
    "summary_embedding": summary_embedding,  # A. 세션 요약 임베딩
    "metadata": {
        "total_speeches": len(raw_df),
        "quality_speeches": len(quality_df),
        "session_summary": session_summary_dict,
        "qa_metrics": qa_metrics_dict
    }
}
db_client.upsert_session_record(session_record)
```

**저장 위치:** `sessions` 테이블

**임베딩 대상:** 세션 요약 텍스트 (raw_summary 또는 key_issues 조합)

---

**단계 2. 안건 항목 저장 (B)**

**저장 시점:** OpenAI API 호출 후 JSON 파싱 직후

**저장 데이터:**
```python
agenda_payload = {
    "session_id": session_id,
    "agenda_title": agenda_title,
    "summary_text": agenda_summary,  # summary_text 또는 consensus_points + conflict_points
    "summary_embedding": agenda_embedding,  # B. 안건별 요약 임베딩
    "metadata": {
        "consensus_points": consensus_points,
        "conflict_points": conflict_points,
        "cooperation_level": cooperation_level
    }
}
db_client.upsert_agenda_items(agenda_rows)
```

**저장 위치:** `agenda_items` 테이블

**임베딩 대상:** 안건 요약 텍스트 (summary_text 또는 consensus_points + conflict_points)

---

**단계 3. 정당 입장 저장 (C)**

**저장 시점:** OpenAI API 호출 후 JSON 파싱 직후

**저장 데이터:**
```python
party_position_row = {
    "agenda_id": agenda_id,
    "party_name": party_name,
    "stance_label": stance_label,
    "key_points": key_points,
    "concerns": concerns,
    "suggestions": suggestions,
    "summary_text": summary_text,  # summary_text 또는 key_points + concerns + suggestions
    "stance_embedding": stance_embedding,  # C. 정당별 입장 임베딩
    "metadata": metadata
}
db_client.upsert_party_positions(party_position_rows)
```

**저장 위치:** `party_positions` 테이블

**임베딩 대상:** 정당 입장 요약 텍스트 (summary_text 또는 key_points + concerns + suggestions)

---

**단계 4. QA 상호작용 저장 (D)**

**저장 시점:** OpenAI API 호출 후 JSON 파싱 직후

**저장 데이터:**
```python
qa_row = {
    "qa_id": uuid4(),
    "agenda_id": agenda_id,
    "questioner": questioner,
    "respondent": answerer,
    "question_text": question,
    "answer_text": answer,
    "question_embedding": question_embedding,  # D. QA 질문 임베딩
    "answer_embedding": answer_embedding,  # D. QA 답변 임베딩
    "metadata": {
        "question_party": question_party,
        "answer_party": answer_party,
        "session_name": session_name
    }
}
db_client.upsert_qa_interactions(qa_rows)
```

**저장 위치:** `qa_interactions` 테이블

**임베딩 대상:** QA 질문 텍스트, QA 답변 텍스트 (배치 처리)

---

### 8-3. E RAG 문서 청킹 및 임베딩 저장

**저장 시점:** A~D 데이터가 Supabase에 저장된 후

**데이터 소스:** A~D의 분석 결과를 재사용하여 RAG 검색용 문서 생성

**단계 1. 세션 요약 RAG 문서 생성 (E-1)**

**데이터 소스:** A (세션 요약 분석 결과)

**처리 과정:**
```python
# 1. 세션 요약에서 텍스트 추출
summary_parts = [
    session_summary.raw_summary,
    "핵심 이슈 요약:",
    *[f"- {issue.issue} ({issue.importance}) : {issue.description}" for issue in key_issues],
    "주요 쟁점:",
    *[f"- {conflict.topic} / 참여 정당: {', '.join(conflict.parties_involved)}" for conflict in major_conflicts],
    "주요 사건:",
    *[f"- {event.event} : {event.description} / 대응 {event.response}" for event in key_events]
]
text = "\n".join(summary_parts)

# 2. 청킹 (800자, 100자 오버랩)
chunks = chunker._split_text(text)

# 3. 각 청크에 임베딩 생성
for chunk in chunks:
    embedding = embedding_client.embed_text(chunk.content)
    
# 4. RAG 문서 저장
rag_document = {
    "source_type": "session_summary",
    "source_id": f"session::{session_name}",
    "chunk_index": chunk_index,
    "content": chunk.content,
    "embedding": embedding,
    "metadata": {
        "session_name": session_name
    }
}
db_client.upsert_rag_documents([rag_document])
```

**저장 위치:** `documents_rag` 테이블

---

**단계 2. 정당별 입장 RAG 문서 생성 (E-2)**

**데이터 소스:** C (정당별 입장 분석 결과)

**처리 과정:**
```python
# 1. 정당별 입장에서 텍스트 추출
text_sections = [
    f"[안건] {agenda_title}",
    f"[정당] {party_name}",
    f"[입장] {stance_label}",
    summary_text,
    "주요 포인트:\n" + "\n".join(f"- {p}" for p in key_points),
    "우려 사항:\n" + "\n".join(f"- {c}" for c in concerns),
    "제안 사항:\n" + "\n".join(f"- {s}" for s in suggestions)
]
text = "\n".join(filter(None, text_sections))

# 2. 청킹 (800자, 100자 오버랩)
chunks = chunker._split_text(text)

# 3. 각 청크에 임베딩 생성
for chunk in chunks:
    embedding = embedding_client.embed_text(chunk.content)
    
# 4. RAG 문서 저장
rag_document = {
    "source_type": "party_position",
    "source_id": f"session::{session_name}::agenda::{agenda_title}::party::{party_name}",
    "chunk_index": chunk_index,
    "content": chunk.content,
    "embedding": embedding,
    "metadata": {
        "session_name": session_name,
        "agenda_title": agenda_title,
        "party_name": party_name,
        "stance_label": stance_label
    }
}
db_client.upsert_rag_documents([rag_document])
```

**저장 위치:** `documents_rag` 테이블

---

**단계 3. QA 페어 RAG 문서 생성 (E-3)**

**데이터 소스:** D (QA 페어 분석 결과)

**처리 과정:**
```python
# 1. QA 페어에서 텍스트 추출
text = (
    f"[질문자] {questioner} ({question_party})\n"
    f"[질문]\n{question}\n\n"
    f"[답변자] {answerer} ({answer_party})\n"
    f"[답변]\n{answer}"
)

# 2. 청킹 (800자, 100자 오버랩)
chunks = chunker._split_text(text)

# 3. 각 청크에 임베딩 생성
for chunk in chunks:
    embedding = embedding_client.embed_text(chunk.content)
    
# 4. RAG 문서 저장
rag_document = {
    "source_type": "qa_pair",
    "source_id": f"session::{session_name}::qa::{index}",
    "chunk_index": chunk_index,
    "content": chunk.content,
    "embedding": embedding,
    "metadata": {
        "session_name": session_name,
        "agenda_title": agenda_title,
        "questioner": questioner,
        "respondent": answerer,
        "effectiveness_bucket": effectiveness_bucket
    }
}
db_client.upsert_rag_documents([rag_document])
```

**저장 위치:** `documents_rag` 테이블

---

### 8-4. 저장 프로세스 요약

| 단계 | 저장 대상 | 데이터 소스 | 저장 위치 | 목적 |
|------|----------|------------|----------|------|
| **1** | 세션 요약 (A) | OpenAI API JSON 응답 | `sessions` 테이블 | 구조화된 데이터 저장 |
| **2** | 안건 요약 (B) | OpenAI API JSON 응답 | `agenda_items` 테이블 | 구조화된 데이터 저장 |
| **3** | 정당 입장 (C) | OpenAI API JSON 응답 | `party_positions` 테이블 | 구조화된 데이터 저장 |
| **4** | QA 페어 (D) | OpenAI API JSON 응답 | `qa_interactions` 테이블 | 구조화된 데이터 저장 |
| **5** | RAG 문서 (E) | A~D 분석 결과 재사용 | `documents_rag` 테이블 | RAG 검색용 벡터 저장 |

**저장 순서의 이유:**
1. **A~D 먼저 저장**: 구조화된 데이터를 먼저 저장하여 외래키 관계 설정
2. **E 나중에 저장**: A~D 데이터를 재사용하여 RAG 검색용 문서 생성
3. **청킹**: 긴 텍스트를 작은 단위로 분할하여 정확한 벡터 검색 지원
4. **메타데이터**: RAG 문서에 source_type, source_id, chunk_index 등 상세 메타데이터 저장

---

### 8-5. 데이터 정합성 보장

**Idempotent 재실행:**
- 기존 세션 데이터 삭제 후 재저장
- hash_digest 및 analysis_version 비교를 통한 변경 감지
- 기존 안건, 정당 입장, QA 데이터 삭제 후 재저장
- 기존 RAG 문서 삭제 후 재저장

**트랜잭션 처리:**
- 세션 레코드 생성 후 session_id 획득
- session_id를 외래 키로 사용하여 관련 데이터 저장
- 실패 시 롤백 가능한 구조

---

## 9장: RAG 문서 청킹 전략

### 9-1. 청킹 대상

**1. 세션 요약 청킹:**
- raw_summary
- key_issues (이슈명, 중요도, 설명)
- major_conflicts (쟁점명, 참여 정당, 성격)
- key_events (사건명, 설명, 대응)

**2. 정당 입장 청킹:**
- 안건명, 정당명, 입장 레이블
- summary_text
- key_points, concerns, suggestions

**3. QA 페어 청킹:**
- 질문자, 답변자 정보
- 질문 텍스트
- 답변 텍스트

### 9-2. 청킹 파라미터

- **청크 크기:** 800자
- **오버랩:** 100자
- **목적:** 의미 단위 유지 및 벡터 검색 최적화

### 9-3. 소스 ID 구조

**세션 요약:**
```
session::{session_name}
```

**정당 입장:**
```
session::{session_name}::agenda::{agenda_title}::party::{party_name}
```

**QA 페어:**
```
session::{session_name}::qa::{index}
```

---

## 10장: 결과 및 활용

### 10-1. 저장된 데이터 활용

**1. 벡터 검색 (Semantic Search):**
- 세션 요약 검색
- 안건별 검색
- 정당 입장 검색
- QA 검색

**2. 분석 대시보드:**
- 세션별 핵심 이슈 시각화
- 정당별 입장 비교
- QA 효과성 분석
- 이슈 트렌드 분석

**3. RAG 시스템:**
- 질의-응답 시스템 구축
- 컨텍스트 기반 답변 생성
- 관련 문서 검색

### 10-2. 확장 가능성

**1. 추가 분석:**
- 이슈 트렌드 분석
- 감정 분석
- 발언자별 분석

**2. 실시간 분석:**
- 새로운 세션 자동 분석
- 변경 사항 감지 및 업데이트

**3. 고급 검색:**
- 하이브리드 검색 (키워드 + 벡터)
- 필터링 검색 (정당, 안건, 기간)

---

## 부록: 기술 스택

### 사용 기술

**프로그래밍 언어:**
- Python 3.x

**라이브러리:**
- OpenAI API (GPT-4, text-embedding-3-small)
- Supabase Python Client
- Pandas (데이터 처리)

**데이터베이스:**
- Supabase (PostgreSQL + Vector Extension)
- pgvector (벡터 검색)

**프로세스:**
- 데이터 전처리
- OpenAI API 호출
- JSON 파싱
- 임베딩 생성
- 데이터베이스 저장

---

## 발표 시 주의사항

1. **프롬프트 예시:** 실제 프롬프트 예시를 보여주며 설명
2. **데이터 흐름:** 각 단계에서 데이터가 어떻게 변환되는지 시각화
3. **테이블 관계:** ERD 다이어그램으로 테이블 관계 설명
4. **임베딩 활용:** 벡터 검색의 장점 및 활용 사례 설명
5. **실제 결과:** 분석 결과 예시를 보여주며 시스템의 효과 설명

