# 중요 발언 선별 개선 방안

## 개선 목표
- 전체 데이터를 기반으로 중요한 발언만 선별
- 의사진행 발언 등 노이즈 제거
- 토큰 제한 내에서 최대한 많은 중요 발언 포함

## 구현 단계

### 1단계: 품질 필터링 강화

**현재 문제:**
- 30자 이상만 체크
- 의사진행 발언 필터링 없음

**개선 방안:**
```python
@staticmethod
def filter_quality_speeches(df: pd.DataFrame) -> pd.DataFrame:
    """품질 발언 필터링 강화."""
    
    # 의사진행 발언 패턴
    procedural_patterns = [
        "의석을 정돈", "회의를 개최", "감사의 말씀", "다음은", "상정합니다",
        "개회를 선포", "폐회를 선포", "의원 여러분", "좋은 말씀", "참석해 주셔서",
        "다음 안건", "이상으로", "다음 순서", "의사진행", "회의 진행"
    ]
    
    def is_valid_speech(row: pd.Series) -> bool:
        text = str(row.get("speech_text", "")).strip()
        
        # 기본 검증
        if pd.isna(text) or not text:
            return False
        if len(text) < 50:  # 최소 길이 증가
            return False
        
        # 의사진행 발언 제외
        text_lower = text.lower()
        if any(pattern in text_lower for pattern in procedural_patterns):
            return False
        
        # 반복 문자 패턴 제외 (예: "감사합니다" 반복)
        if len(set(text[:20])) < 5:  # 처음 20자가 너무 단순하면 제외
            return False
        
        return True
    
    quality_mask = df.apply(is_valid_speech, axis=1)
    return df[quality_mask].copy()
```

### 2단계: 중요도 점수 계산 함수 추가

```python
@staticmethod
def calculate_importance_score(row: pd.Series) -> float:
    """발언의 중요도 점수 계산 (0-1)."""
    text = str(row.get("speech_text", "")).strip()
    text_lower = text.lower()
    
    score = 0.0
    
    # 1. 길이 점수 (적당한 길이가 중요)
    length = len(text)
    if 100 <= length <= 1000:
        score += 0.2
    elif 50 <= length < 100 or 1000 < length <= 2000:
        score += 0.1
    
    # 2. 정책 키워드 점수
    policy_keywords = [
        "정책", "법안", "예산", "제안", "건의", "개선", "문제", "해결",
        "국민", "사회", "경제", "복지", "교육", "보건", "환경", "안전",
        "비판", "지적", "우려", "필요", "중요", "시급"
    ]
    keyword_count = sum(1 for keyword in policy_keywords if keyword in text_lower)
    score += min(keyword_count * 0.1, 0.4)  # 최대 0.4점
    
    # 3. 질문/답변 패턴 점수
    qa_patterns = ["질의", "질문", "답변", "설명", "문의", "알고 싶"]
    if any(pattern in text_lower for pattern in qa_patterns):
        score += 0.2
    
    # 4. 숫자/통계 포함 점수 (구체적인 내용)
    import re
    if re.search(r'\d+[억만천]', text) or re.search(r'\d+%', text):
        score += 0.1
    
    # 5. 문장 복잡도 점수 (단순 반복이 아닌 실제 내용)
    sentences = text.split('。') + text.split('.') + text.split('!') + text.split('?')
    if len([s for s in sentences if len(s.strip()) > 20]) >= 3:
        score += 0.1
    
    return min(score, 1.0)  # 최대 1.0
```

### 3단계: 중요도 기반 샘플링 함수

```python
@staticmethod
def select_important_speeches(
    df: pd.DataFrame,
    *,
    max_speeches: int = 50,
    min_importance: float = 0.3,
    party_balance: bool = True
) -> pd.DataFrame:
    """중요도 기반으로 발언 선별."""
    
    # 중요도 점수 계산
    df = df.copy()
    df['importance_score'] = df.apply(
        lambda row: SessionAnalysisWorkflow.calculate_importance_score(row),
        axis=1
    )
    
    # 최소 중요도 이상만 선택
    df = df[df['importance_score'] >= min_importance].copy()
    
    if party_balance:
        # 정당별 균형 유지하면서 중요도 높은 것 선택
        selected = []
        parties = df['party'].dropna().unique()
        speeches_per_party = max(1, max_speeches // len(parties))
        
        for party in parties:
            party_speeches = df[df['party'] == party].copy()
            party_speeches = party_speeches.sort_values(
                'importance_score', ascending=False
            )
            selected.append(party_speeches.head(speeches_per_party))
        
        result = pd.concat(selected, ignore_index=True)
        # 전체에서도 중요도 순으로 추가 선택
        remaining = max_speeches - len(result)
        if remaining > 0:
            all_speeches = df[~df.index.isin(result.index)].copy()
            all_speeches = all_speeches.sort_values(
                'importance_score', ascending=False
            )
            result = pd.concat([result, all_speeches.head(remaining)], ignore_index=True)
        
        return result.head(max_speeches)
    else:
        # 단순 중요도 순 정렬
        df = df.sort_values('importance_score', ascending=False)
        return df.head(max_speeches)
```

### 4단계: prepare_session_summary_payload 수정

```python
def prepare_session_summary_payload(
    self, 
    df: pd.DataFrame,
    *,
    max_sample_speeches: int = 50,
    max_chars_per_speech: int = 500
) -> Dict[str, Any]:
    """회차 요약을 위한 데이터 준비 (중요도 기반 선별)."""
    
    # 통계 계산
    agenda_stats: Dict[str, int] = {}
    for agenda in df["agenda_item_titles"].dropna().unique():
        if pd.notna(agenda) and str(agenda).strip():
            agenda_count = len(df[df["agenda_item_titles"] == agenda])
            agenda_stats[str(agenda)] = agenda_count
    
    party_stats = df["party"].value_counts().to_dict()
    
    # 중요도 기반 선별
    important_speeches = self.select_important_speeches(
        df,
        max_speeches=max_sample_speeches,
        min_importance=0.3,
        party_balance=True
    )
    
    # 샘플 데이터 구성
    speeches_sample: List[Dict[str, Any]] = []
    for _, row in important_speeches.iterrows():
        speeches_sample.append({
            "party": row.get("party", ""),
            "speaker": row.get("speaker_name", ""),
            "text": str(row.get("speech_text", ""))[:max_chars_per_speech],
            "importance_score": row.get("importance_score", 0.0),
        })
    
    return {
        "total_speeches": len(df),
        "quality_speeches": len(important_speeches),
        "agenda_stats": agenda_stats,
        "party_stats": party_stats,
        "speeches_sample": speeches_sample,
    }
```

### 5단계: prepare_agenda_payloads 수정

```python
def prepare_agenda_payloads(
    self, 
    df: pd.DataFrame, 
    *,
    top_agendas: int = 3,
    max_speeches_per_party: int = 10
) -> List[Dict[str, Any]]:
    """안건별 발언을 LLM 프롬프트에 맞게 정리 (중요도 기반)."""
    
    agenda_counts = df["agenda_item_titles"].value_counts()
    top_agenda_titles = [
        title for title in agenda_counts.head(top_agendas).index.tolist() 
        if pd.notna(title)
    ]
    
    payloads: List[Dict[str, Any]] = []
    for agenda_title in top_agenda_titles:
        if not str(agenda_title).strip():
            continue
        
        agenda_df = df[df["agenda_item_titles"] == agenda_title].copy()
        
        # 중요도 점수 계산
        agenda_df['importance_score'] = agenda_df.apply(
            lambda row: self.calculate_importance_score(row),
            axis=1
        )
        
        party_speeches: Dict[str, List[str]] = {}
        for party in agenda_df["party"].dropna().unique():
            party_data = agenda_df[agenda_df["party"] == party].copy()
            # 중요도 순 정렬
            party_data = party_data.sort_values(
                'importance_score', ascending=False
            )
            # 상위 N개 선택
            selected = party_data.head(max_speeches_per_party)
            party_speeches[str(party)] = [
                str(row.get("speech_text", ""))[:500] 
                for _, row in selected.iterrows()
            ]
        
        payloads.append({
            "agenda_title": str(agenda_title),
            "total_speeches": len(agenda_df),
            "party_speeches": party_speeches,
        })
    
    return payloads
```

## 토큰 제한 고려사항

### 토큰 계산
- 한국어: 약 1.3자 = 1 토큰
- GPT-4: 약 128K 토큰 (약 96,000자)
- 안전 마진: 약 80,000자까지 사용

### 최적화 전략
1. **동적 길이 조절:**
   - 발언 수가 많으면 각 발언 길이 제한 (300자)
   - 발언 수가 적으면 각 발언 길이 증가 (500자)

2. **중요도 기반 길이 조절:**
   - 중요도 높은 발언: 더 긴 텍스트 포함
   - 중요도 낮은 발언: 짧게 제한

3. **단계적 포함:**
   - 최소 중요도 기준을 높여가며 발언 수 조절
   - 토큰 제한에 도달하면 중단

## 테스트 및 검증

1. **샘플링 품질 검증:**
   - 선별된 발언이 실제로 중요한지 수동 검토
   - 의사진행 발언이 포함되지 않았는지 확인

2. **분석 품질 비교:**
   - 기존 랜덤 샘플링 vs 중요도 기반 샘플링
   - 분석 결과의 정확도 비교

3. **성능 테스트:**
   - 토큰 사용량 모니터링
   - 처리 시간 측정

## 구현 우선순위

1. **1단계 (필수):** 품질 필터링 강화
2. **2단계 (필수):** 중요도 점수 계산
3. **3단계 (권장):** 중요도 기반 샘플링
4. **4-5단계 (선택):** 함수 수정 및 통합


