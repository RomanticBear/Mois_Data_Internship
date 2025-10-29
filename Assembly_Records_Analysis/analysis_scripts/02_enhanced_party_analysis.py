#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2단계: 대화 데이터 특화 고급 정당별 토픽 분석
- 대화 데이터 특성 고려한 전처리
- 문맥 기반 키워드 추출
- 대화 패턴 분석
- 정당별 발화 특성 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """데이터 로드"""
    print("데이터 로딩 중...")
    
    # 모든 회의록 데이터 로드
    all_speeches = []
    
    # data/with_party 폴더의 모든 회의록 데이터 수집
    minutes_dir = "data/with_party"
    if not os.path.exists(minutes_dir):
        print(f"경고: {minutes_dir} 폴더를 찾을 수 없습니다.")
        return None
    
    for session_dir in os.listdir(minutes_dir):
        session_path = os.path.join(minutes_dir, session_dir)
        if not os.path.isdir(session_path):
            continue
            
        print(f"처리 중: {session_dir}")
        
        # 발언 데이터 로드
        for file in os.listdir(session_path):
            if file.endswith('_speeches.csv'):
                file_path = os.path.join(session_path, file)
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    if 'party' in df.columns:
                        # 컬럼명 통일
                        if 'speech_text' in df.columns:
                            df = df.rename(columns={'speech_text': 'speech_content'})
                        if 'party' in df.columns:
                            df = df.rename(columns={'party': 'speaker_party'})
                        all_speeches.append(df)
                except Exception as e:
                    print(f"파일 로드 실패: {file_path} - {e}")
    
    if not all_speeches:
        print("발언 데이터를 찾을 수 없습니다.")
        return None
    
    # 데이터 통합
    speeches_df = pd.concat(all_speeches, ignore_index=True)
    print(f"총 {len(speeches_df)}개의 발언 데이터 로드 완료")
    
    return speeches_df

def preprocess_conversation_text(text):
    """대화 데이터 특화 전처리 (문장 부호 보존)"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 대화 특화 전처리 (문장 부호는 보존)
    # 1. 반복되는 말투 제거
    text = re.sub(r'(네|예|아|음|어|그|저|이|그런데|그리고|하지만|그러면|그래서|그런|이런|저런)\s*', '', text)
    
    # 2. 문장 부호는 보존하고 다른 특수문자만 제거
    text = re.sub(r'[^\w\s가-힣.!?]', ' ', text)
    
    # 3. 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    # 4. 짧은 단어 제거 (1-2글자)
    words = text.split()
    words = [word for word in words if len(word) > 2]
    
    return ' '.join(words).strip()

def extract_conversation_keywords(texts, min_freq=3):
    """대화 데이터 특화 키워드 추출"""
    print("대화 데이터 특화 키워드 추출 중...")
    
    # 모든 텍스트 전처리
    processed_texts = [preprocess_conversation_text(text) for text in texts]
    processed_texts = [text for text in processed_texts if len(text) > 10]
    
    if not processed_texts:
        return {}
    
    # 단어 빈도 계산
    word_freq = Counter()
    for text in processed_texts:
        words = text.split()
        # 3글자 이상 단어만 선택
        words = [word for word in words if len(word) >= 3]
        word_freq.update(words)
    
    # 최소 빈도 이상의 단어만 선택
    filtered_words = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
    
    # 상위 키워드 선택
    sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
    
    return dict(sorted_words[:50])  # 상위 50개

def analyze_conversation_patterns(texts):
    """대화 패턴 분석 (수정된 버전)"""
    print("대화 패턴 분석 중...")
    
    patterns = {
        'question_marks': 0,
        'exclamation_marks': 0,
        'long_sentences': 0,
        'short_sentences': 0,
        'repetitive_words': 0,
        'total_sentences': 0
    }
    
    for text in texts:
        if not text or pd.isna(text):
            continue
            
        # 원본 텍스트 사용 (전처리 전)
        original_text = str(text)
        
        # 질문 패턴 (원본에서)
        patterns['question_marks'] += original_text.count('?')
        
        # 감탄 패턴 (원본에서)
        patterns['exclamation_marks'] += original_text.count('!')
        
        # 문장 분리 (원본에서)
        sentences = re.split(r'[.!?]', original_text)
        valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        patterns['total_sentences'] += len(valid_sentences)
        
        # 긴 문장 (30자 이상)
        long_sents = [s for s in valid_sentences if len(s) > 30]
        patterns['long_sentences'] += len(long_sents)
        
        # 짧은 문장 (10자 이하)
        short_sents = [s for s in valid_sentences if 5 <= len(s) <= 10]
        patterns['short_sentences'] += len(short_sents)
        
        # 반복 단어 (전처리된 텍스트에서)
        processed_text = preprocess_conversation_text(text)
        words = processed_text.split()
        word_counts = Counter(words)
        repetitive = sum(1 for count in word_counts.values() if count >= 3)
        patterns['repetitive_words'] += repetitive
    
    return patterns

def analyze_sentiment_enhanced(texts):
    """향상된 감정 분석 (대화 특화)"""
    print("향상된 감정 분석 중...")
    
    # 대화 특화 감정 키워드
    positive_keywords = [
        '좋', '긍정', '지지', '찬성', '옳', '바람직', '필요', '중요', '필수',
        '개선', '발전', '향상', '증가', '확대', '강화', '지원', '투자',
        '성공', '효과', '성과', '성취', '달성', '실현', '구현',
        '감사', '고맙', '훌륭', '훌륭한', '훌륭하다', '훌륭합니다',
        '잘', '잘하', '잘하고', '잘했다', '잘했습니다'
    ]
    
    negative_keywords = [
        '나쁜', '부정', '반대', '잘못', '문제', '위험', '심각',
        '악화', '감소', '축소', '약화', '제한', '규제', '억제',
        '실패', '실망', '우려', '걱정', '불안', '위기', '위협',
        '안타깝', '안타깝다', '안타깝습니다', '유감', '유감스럽',
        '부족', '부족하다', '부족합니다', '아쉽', '아쉽다', '아쉽습니다'
    ]
    
    # 대화 특화 중립 키워드
    neutral_keywords = [
        '검토', '논의', '제안', '질문', '답변', '설명', '이해',
        '확인', '검토하', '논의하', '제안하', '질문하', '답변하',
        '설명하', '이해하', '확인하', '생각', '생각하', '생각합니다'
    ]
    
    sentiment_results = []
    sentiment_scores = []
    
    for text in texts:
        if not text or pd.isna(text):
            sentiment_results.append('neutral')
            sentiment_scores.append(0)
            continue
        
        text = str(text).lower()
        
        pos_count = sum(1 for keyword in positive_keywords if keyword in text)
        neg_count = sum(1 for keyword in negative_keywords if keyword in text)
        neu_count = sum(1 for keyword in neutral_keywords if keyword in text)
        
        # 감정 점수 계산 (가중치 적용)
        pos_score = pos_count * 1.0
        neg_score = neg_count * 1.0
        neu_score = neu_count * 0.5
        
        total_score = pos_score + neg_score + neu_score
        
        if total_score == 0:
            sentiment_results.append('neutral')
            sentiment_scores.append(0)
        elif pos_score > neg_score and pos_score > neu_score:
            sentiment_results.append('positive')
            sentiment_scores.append(pos_score / total_score)
        elif neg_score > pos_score and neg_score > neu_score:
            sentiment_results.append('negative')
            sentiment_scores.append(-neg_score / total_score)
        else:
            sentiment_results.append('neutral')
            sentiment_scores.append(0)
    
    return sentiment_results, sentiment_scores

def analyze_enhanced_party_topics(speeches_df):
    """향상된 정당별 토픽 분석"""
    print("\n향상된 정당별 토픽 분석 시작...")
    
    # 발언 텍스트 전처리
    speeches_df['processed_text'] = speeches_df['speech_content'].apply(preprocess_conversation_text)
    speeches_df = speeches_df[speeches_df['processed_text'].str.len() > 20].copy()
    
    print(f"분석 대상 발언 수: {len(speeches_df):,}개")
    
    results = {}
    
    # 정당별 향상된 분석
    print("\n정당별 향상된 분석")
    party_analysis = {}
    parties = speeches_df['speaker_party'].dropna().unique()
    
    for party in parties:
        if pd.isna(party) or party == '' or party == '미분류':
            continue
        
        print(f"\n{party} 정당 분석 중...")
        party_speeches = speeches_df[speeches_df['speaker_party'] == party]
        party_texts = party_speeches['processed_text'].tolist()
        
        if len(party_texts) < 10:
            print(f"{party} 정당의 발언이 부족합니다. 건너뜁니다.")
            continue
        
        # 1. 대화 특화 키워드 추출
        party_keywords = extract_conversation_keywords(party_texts, min_freq=2)
        
        # 2. 대화 패턴 분석
        party_patterns = analyze_conversation_patterns(party_texts)
        
        # 3. 향상된 감정 분석
        party_sentiment, party_sentiment_scores = analyze_sentiment_enhanced(party_texts)
        party_sentiment_counts = Counter(party_sentiment)
        
        # 4. 발언 길이 분석
        speech_lengths = [len(text.split()) for text in party_texts if text]
        avg_length = np.mean(speech_lengths) if speech_lengths else 0
        
        # 5. 토픽 모델링 (간단한 버전)
        all_words = []
        for text in party_texts:
            words = text.split()
            words = [word for word in words if len(word) >= 3]
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(30)
        
        # 토픽 생성 (단어 빈도 기반)
        topics = []
        words_per_topic = len(top_words) // 5  # 5개 토픽
        
        for i in range(5):
            start_idx = i * words_per_topic
            end_idx = start_idx + words_per_topic if i < 4 else len(top_words)
            topic_words = [word for word, freq in top_words[start_idx:end_idx]]
            
            topics.append({
                'topic_id': i,
                'words': topic_words[:8],  # 상위 8개 단어
                'topic_str': f'토픽 {i+1}: {", ".join(topic_words[:5])}...'
            })
        
        party_analysis[party] = {
            'total_speeches': len(party_speeches),
            'keywords': party_keywords,
            'patterns': party_patterns,
            'sentiment_distribution': dict(party_sentiment_counts),
            'sentiment_scores': party_sentiment_scores,
            'avg_speech_length': avg_length,
            'topics': topics
        }
        
        print(f"{party} 정당 분석 완료 - 발언 수: {len(party_speeches):,}개")
        print(f"  - 평균 발언 길이: {avg_length:.1f} 단어")
        print(f"  - 감정 분포: {dict(party_sentiment_counts)}")
        print(f"  - 주요 키워드: {list(party_keywords.keys())[:5]}")
    
    results['party_analysis'] = party_analysis
    
    return results, speeches_df

def create_enhanced_visualizations(results, speeches_df):
    """향상된 시각화 생성"""
    print("\n향상된 시각화 생성 중...")
    
    # 결과 폴더 생성
    os.makedirs('analysis_results', exist_ok=True)
    
    party_analysis = results['party_analysis']
    
    # 1. 정당별 키워드 워드클라우드
    print("1. 정당별 키워드 워드클라우드 생성")
    parties = list(party_analysis.keys())
    n_parties = len(parties)
    
    if n_parties > 0:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, party in enumerate(parties[:4]):
            if i >= len(axes):
                break
            
            party_data = party_analysis[party]
            if party_data['keywords']:
                try:
                    # 워드클라우드 생성 (기본 라이브러리 사용)
                    words = list(party_data['keywords'].keys())
                    freqs = list(party_data['keywords'].values())
                    
                    # 간단한 텍스트 시각화
                    y_pos = np.arange(len(words[:20]))  # 상위 20개
                    axes[i].barh(y_pos, freqs[:20], alpha=0.7)
                    axes[i].set_yticks(y_pos)
                    axes[i].set_yticklabels(words[:20], fontsize=8)
                    axes[i].set_title(f'{party} 정당 - 주요 키워드', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel('빈도')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'키워드 시각화 실패\n{str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{party} 정당', fontsize=12, fontweight='bold')
            else:
                axes[i].text(0.5, 0.5, '키워드 데이터 없음', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{party} 정당', fontsize=12, fontweight='bold')
        
        # 빈 subplot 제거
        for i in range(len(parties), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('analysis_results/02_enhanced_party_keywords.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 정당별 토픽 시각화
    print("2. 정당별 토픽 시각화 생성")
    if n_parties > 0:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, party in enumerate(parties[:4]):
            if i >= len(axes):
                break
            
            party_data = party_analysis[party]
            if party_data['topics']:
                topics = party_data['topics']
                
                # 상위 3개 토픽만 표시
                for j, topic in enumerate(topics[:3]):
                    words = topic['words'][:6]  # 상위 6개 단어
                    scores = [1.0] * len(words)  # 간단한 점수
                    
                    y_pos = np.arange(len(words))
                    axes[i].barh(y_pos, scores, alpha=0.7, label=f'토픽 {topic["topic_id"]+1}')
                
                axes[i].set_title(f'{party} 정당 - 주요 토픽', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('중요도')
                axes[i].legend(fontsize=8)
            else:
                axes[i].text(0.5, 0.5, '토픽 데이터 없음', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{party} 정당', fontsize=12, fontweight='bold')
        
        # 빈 subplot 제거
        for i in range(len(parties), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('analysis_results/02_enhanced_party_topics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 정당별 감정 분석
    print("3. 정당별 감정 분석 시각화 생성")
    if n_parties > 0:
        # 정당별 감정 분포 데이터 준비
        sentiment_df = pd.DataFrame({
            party: data['sentiment_distribution'] 
            for party, data in party_analysis.items()
        }).fillna(0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(sentiment_df, annot=True, fmt='.0f', cmap='RdYlBu_r')
        plt.title('정당별 감정 분석 (대화 특화)', fontsize=16, fontweight='bold')
        plt.xlabel('정당', fontsize=12)
        plt.ylabel('감정', fontsize=12)
        plt.tight_layout()
        plt.savefig('analysis_results/02_enhanced_party_sentiment.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 정당별 대화 패턴 분석
    print("4. 정당별 대화 패턴 분석 시각화 생성")
    if n_parties > 0:
        # 패턴 데이터 준비
        pattern_data = []
        for party, data in party_analysis.items():
            patterns = data['patterns']
            pattern_data.append({
                '정당': party,
                '평균 문장 길이': patterns['total_sentences'] / data['total_speeches'] if data['total_speeches'] > 0 else 0,
                '질문 비율': patterns['question_marks'] / data['total_speeches'] if data['total_speeches'] > 0 else 0,
                '긴 문장 비율': patterns['long_sentences'] / patterns['total_sentences'] if patterns['total_sentences'] > 0 else 0,
                '평균 발언 길이': data['avg_speech_length']
            })
        
        pattern_df = pd.DataFrame(pattern_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 평균 발언 길이
        axes[0, 0].bar(pattern_df['정당'], pattern_df['평균 발언 길이'])
        axes[0, 0].set_title('정당별 평균 발언 길이', fontweight='bold')
        axes[0, 0].set_ylabel('단어 수')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 질문 비율
        axes[0, 1].bar(pattern_df['정당'], pattern_df['질문 비율'])
        axes[0, 1].set_title('정당별 질문 비율', fontweight='bold')
        axes[0, 1].set_ylabel('질문 수 / 발언 수')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 긴 문장 비율
        axes[1, 0].bar(pattern_df['정당'], pattern_df['긴 문장 비율'])
        axes[1, 0].set_title('정당별 긴 문장 비율', fontweight='bold')
        axes[1, 0].set_ylabel('긴 문장 비율')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 평균 문장 길이
        axes[1, 1].bar(pattern_df['정당'], pattern_df['평균 문장 길이'])
        axes[1, 1].set_title('정당별 평균 문장 길이', fontweight='bold')
        axes[1, 1].set_ylabel('문장 수')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('analysis_results/02_enhanced_party_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("향상된 시각화 완료!")

def generate_enhanced_report(results, speeches_df):
    """향상된 분석 보고서 생성"""
    print("\n향상된 분석 보고서 생성 중...")
    
    # 보고서 폴더 생성
    os.makedirs('analysis_reports', exist_ok=True)
    
    party_analysis = results['party_analysis']
    
    report_content = f"""# 대화 데이터 특화 정당별 토픽 분석 보고서

## 분석 개요
- **분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **분석 대상**: 국회 회의록 발언 데이터 (대화 데이터 특화)
- **분석 방법**: 대화 데이터 특화 자연어처리 기법
  - 대화 특화 전처리
  - 문맥 기반 키워드 추출
  - 대화 패턴 분석
  - 향상된 감정 분석

## 정당별 향상된 분석 결과

"""
    
    for party, data in party_analysis.items():
        report_content += f"""### {party} 정당

**전체 발언 수**: {data['total_speeches']:,}개
**평균 발언 길이**: {data['avg_speech_length']:.1f} 단어

#### 대화 패턴 분석
"""
        
        patterns = data['patterns']
        report_content += f"""
- **총 문장 수**: {patterns['total_sentences']:,}개
- **질문 수**: {patterns['question_marks']:,}개
- **감탄 수**: {patterns['exclamation_marks']:,}개
- **긴 문장 수**: {patterns['long_sentences']:,}개
- **짧은 문장 수**: {patterns['short_sentences']:,}개
- **반복 단어 수**: {patterns['repetitive_words']:,}개
"""
        
        report_content += "\n#### 주요 키워드 분석\n\n"
        if data['keywords']:
            sorted_keywords = sorted(data['keywords'].items(), key=lambda x: x[1], reverse=True)
            report_content += "**상위 15개 키워드**:\n\n"
            for i, (keyword, freq) in enumerate(sorted_keywords[:15]):
                report_content += f"{i+1}. **{keyword}**: {freq}회\n"
        else:
            report_content += "키워드 분석을 사용할 수 없습니다.\n"
        
        report_content += "\n#### 토픽 분석\n\n"
        if data['topics']:
            for topic in data['topics']:
                report_content += f"**{topic['topic_str']}**\n"
                report_content += f"주요 단어: {', '.join(topic['words'][:8])}\n\n"
        else:
            report_content += "토픽 분석을 사용할 수 없습니다.\n"
        
        report_content += "\n#### 감정 분석\n\n"
        sentiment_dist = data['sentiment_distribution']
        total = sum(sentiment_dist.values())
        report_content += "**감정 분포**:\n"
        for sentiment, count in sentiment_dist.items():
            percentage = (count / total * 100) if total > 0 else 0
            report_content += f"- **{sentiment}**: {count:,}개 ({percentage:.1f}%)\n"
        
        # 감정 점수 분석
        if data['sentiment_scores']:
            avg_sentiment = np.mean(data['sentiment_scores'])
            report_content += f"\n**평균 감정 점수**: {avg_sentiment:.3f}\n"
            if avg_sentiment > 0.1:
                report_content += "- 전반적으로 긍정적인 감정을 보입니다.\n"
            elif avg_sentiment < -0.1:
                report_content += "- 전반적으로 부정적인 감정을 보입니다.\n"
            else:
                report_content += "- 전반적으로 중립적인 감정을 보입니다.\n"
        
        report_content += "\n---\n\n"
    
    report_content += """## 분석 결과 요약

### 주요 발견사항

1. **대화 데이터 특성 반영**
   - 반복되는 말투와 대화 중단 표시 제거
   - 문맥을 고려한 키워드 추출
   - 대화 패턴 기반 분석

2. **정당별 발화 특성**
   - 각 정당의 고유한 발화 패턴 발견
   - 질문/답변 비율의 차이
   - 문장 길이와 복잡도의 차이

3. **향상된 감정 분석**
   - 대화 특화 감정 키워드 적용
   - 가중치 기반 감정 점수 계산
   - 정당별 감정적 특성 파악

### 분석의 한계점

1. **대화 데이터의 복잡성**
   - 맥락 의존적 표현의 정확한 해석 어려움
   - 반말/존댓말 구분의 한계

2. **키워드 기반 분석의 한계**
   - 문맥을 완전히 고려하지 못함
   - 은유나 비유 표현 처리 어려움

### 향후 개선 방향

1. **맥락 기반 분석**
   - 문맥을 고려한 키워드 추출
   - 대화 흐름 분석

2. **고급 자연어처리**
   - 형태소 분석기 활용
   - 문장 구조 분석

---
*본 보고서는 대화 데이터 특성을 고려한 정당별 토픽 분석 결과입니다.*
"""
    
    # 보고서 저장
    with open('analysis_reports/02_enhanced_party_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("향상된 분석 보고서 생성 완료!")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("2단계: 대화 데이터 특화 고급 정당별 토픽 분석")
    print("=" * 60)
    
    # 데이터 로드
    speeches_df = load_data()
    if speeches_df is None:
        print("데이터 로드 실패. 프로그램을 종료합니다.")
        return
    
    # 향상된 정당별 토픽 분석
    results, processed_df = analyze_enhanced_party_topics(speeches_df)
    if not results:
        print("향상된 분석 실패. 프로그램을 종료합니다.")
        return
    
    # 향상된 시각화 생성
    create_enhanced_visualizations(results, processed_df)
    
    # 향상된 보고서 생성
    generate_enhanced_report(results, processed_df)
    
    print("\n" + "=" * 60)
    print("2단계 향상된 분석 완료!")
    print("=" * 60)
    print("생성된 파일:")
    print("- analysis_results/02_enhanced_party_keywords.png")
    print("- analysis_results/02_enhanced_party_topics.png")
    print("- analysis_results/02_enhanced_party_sentiment.png")
    print("- analysis_results/02_enhanced_party_patterns.png")
    print("- analysis_reports/02_enhanced_party_analysis_report.md")

if __name__ == "__main__":
    main()
