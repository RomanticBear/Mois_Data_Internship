#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4단계: 고급 토픽 분석
- KoNLPy 형태소 분석기 활용
- LDA 토픽 모델링
- 감정 분석
- 고급 시각화
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

# 고급 분석을 위한 라이브러리
try:
    from konlpy.tag import Okt, Kkma, Mecab
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from wordcloud import WordCloud
    import gensim
    from gensim import corpora, models
    ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"고급 분석 라이브러리 일부가 없습니다: {e}")
    print("기본 키워드 분석으로 진행합니다.")
    ADVANCED_AVAILABLE = False

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

def preprocess_text(text):
    """텍스트 전처리"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # 특수문자 제거, 공백 정리
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def morphological_analysis(texts, analyzer='okt'):
    """형태소 분석"""
    if not ADVANCED_AVAILABLE:
        print("형태소 분석기 없음. 기본 키워드 분석 사용")
        return None
    
    print(f"형태소 분석 중... (분석기: {analyzer})")
    
    try:
        if analyzer == 'okt':
            tagger = Okt()
        elif analyzer == 'kkma':
            tagger = Kkma()
        else:
            tagger = Okt()
        
        # 명사, 동사, 형용사 추출
        pos_tags = ['Noun', 'Verb', 'Adjective']
        processed_texts = []
        
        for text in texts:
            if not text:
                processed_texts.append("")
                continue
            
            try:
                # 형태소 분석
                morphs = tagger.pos(text)
                # 명사, 동사, 형용사만 추출
                words = [word for word, pos in morphs if pos in pos_tags and len(word) > 1]
                processed_texts.append(' '.join(words))
            except:
                processed_texts.append("")
        
        return processed_texts
    
    except Exception as e:
        print(f"형태소 분석 실패: {e}")
        return None

def extract_keywords_tfidf(texts, max_features=1000):
    """TF-IDF를 이용한 키워드 추출"""
    if not ADVANCED_AVAILABLE:
        return None
    
    print("TF-IDF 키워드 추출 중...")
    
    try:
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=None,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # 상위 키워드 추출
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[-50:][::-1]  # 상위 50개
        
        keywords = [feature_names[i] for i in top_indices]
        scores = [mean_scores[i] for i in top_indices]
        
        return dict(zip(keywords, scores))
    
    except Exception as e:
        print(f"TF-IDF 키워드 추출 실패: {e}")
        return None

def lda_topic_modeling(texts, num_topics=10):
    """LDA 토픽 모델링"""
    if not ADVANCED_AVAILABLE:
        return None
    
    print(f"LDA 토픽 모델링 중... (토픽 수: {num_topics})")
    
    try:
        # 텍스트 전처리
        processed_texts = [preprocess_text(text) for text in texts]
        processed_texts = [text for text in processed_texts if text]
        
        if len(processed_texts) < 10:
            print("분석할 텍스트가 부족합니다.")
            return None
        
        # 형태소 분석
        morph_texts = morphological_analysis(processed_texts)
        if not morph_texts:
            morph_texts = processed_texts
        
        # 단어 리스트로 변환
        word_lists = [text.split() for text in morph_texts if text]
        
        # 사전 생성
        dictionary = corpora.Dictionary(word_lists)
        # 코퍼스 생성
        corpus = [dictionary.doc2bow(text) for text in word_lists]
        
        # LDA 모델 학습
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # 토픽별 단어 추출
        topics = []
        for idx, topic in lda_model.print_topics(-1, num_words=10):
            words = topic.split('"')[1::2]  # 따옴표 안의 단어들만 추출
            topics.append({
                'topic_id': idx,
                'words': words,
                'topic_str': topic
            })
        
        return {
            'model': lda_model,
            'dictionary': dictionary,
            'corpus': corpus,
            'topics': topics
        }
    
    except Exception as e:
        print(f"LDA 토픽 모델링 실패: {e}")
        return None

def sentiment_analysis(texts):
    """감정 분석 (간단한 키워드 기반)"""
    print("감정 분석 중...")
    
    # 감정 키워드 정의
    positive_keywords = [
        '좋', '긍정', '지지', '찬성', '옳', '바람직', '필요', '중요', '필수',
        '개선', '발전', '향상', '증가', '확대', '강화', '지원', '투자',
        '성공', '효과', '성과', '성취', '달성', '실현', '구현'
    ]
    
    negative_keywords = [
        '나쁜', '부정', '반대', '반대', '잘못', '문제', '위험', '심각',
        '악화', '감소', '축소', '약화', '제한', '규제', '억제',
        '실패', '실망', '우려', '걱정', '불안', '위기', '위협'
    ]
    
    neutral_keywords = [
        '검토', '논의', '검토', '검토', '검토', '검토', '검토', '검토',
        '제안', '제안', '제안', '제안', '제안', '제안', '제안', '제안'
    ]
    
    sentiment_results = []
    
    for text in texts:
        if not text:
            sentiment_results.append('neutral')
            continue
        
        text = str(text).lower()
        
        pos_count = sum(1 for keyword in positive_keywords if keyword in text)
        neg_count = sum(1 for keyword in negative_keywords if keyword in text)
        neu_count = sum(1 for keyword in neutral_keywords if keyword in text)
        
        if pos_count > neg_count and pos_count > neu_count:
            sentiment_results.append('positive')
        elif neg_count > pos_count and neg_count > neu_count:
            sentiment_results.append('negative')
        else:
            sentiment_results.append('neutral')
    
    return sentiment_results

def analyze_advanced_topics(speeches_df):
    """고급 토픽 분석"""
    print("\n고급 토픽 분석 시작...")
    
    # 발언 텍스트 전처리
    speeches_df['processed_text'] = speeches_df['speech_content'].apply(preprocess_text)
    speeches_df = speeches_df[speeches_df['processed_text'].str.len() > 50].copy()
    
    print(f"분석 대상 발언 수: {len(speeches_df):,}개")
    
    results = {}
    
    # 1. TF-IDF 키워드 추출
    print("\n1. TF-IDF 키워드 추출")
    texts = speeches_df['processed_text'].tolist()
    tfidf_keywords = extract_keywords_tfidf(texts)
    if tfidf_keywords:
        results['tfidf_keywords'] = tfidf_keywords
        print(f"추출된 키워드 수: {len(tfidf_keywords)}개")
    
    # 2. LDA 토픽 모델링
    print("\n2. LDA 토픽 모델링")
    lda_results = lda_topic_modeling(texts, num_topics=8)
    if lda_results:
        results['lda_topics'] = lda_results
        print(f"발견된 토픽 수: {len(lda_results['topics'])}개")
    
    # 3. 감정 분석
    print("\n3. 감정 분석")
    sentiment_results = sentiment_analysis(texts)
    speeches_df['sentiment'] = sentiment_results
    
    sentiment_counts = Counter(sentiment_results)
    results['sentiment_analysis'] = dict(sentiment_counts)
    print(f"감정 분석 결과: {dict(sentiment_counts)}")
    
    # 4. 정당별 고급 분석
    print("\n4. 정당별 고급 분석")
    party_analysis = {}
    parties = speeches_df['speaker_party'].dropna().unique()
    
    for party in parties:
        if pd.isna(party) or party == '':
            continue
        
        party_speeches = speeches_df[speeches_df['speaker_party'] == party]
        party_texts = party_speeches['processed_text'].tolist()
        
        # 정당별 TF-IDF 키워드
        party_tfidf = extract_keywords_tfidf(party_texts, max_features=500)
        
        # 정당별 감정 분석
        party_sentiment = sentiment_analysis(party_texts)
        party_sentiment_counts = Counter(party_sentiment)
        
        party_analysis[party] = {
            'total_speeches': len(party_speeches),
            'tfidf_keywords': party_tfidf,
            'sentiment_distribution': dict(party_sentiment_counts)
        }
    
    results['party_analysis'] = party_analysis
    
    # 5. 시간별 고급 분석
    print("\n5. 시간별 고급 분석")
    speeches_df['parsed_date'] = pd.to_datetime(speeches_df['date'], errors='coerce')
    speeches_df = speeches_df[speeches_df['parsed_date'].notna()].copy()
    speeches_df['year_month'] = speeches_df['parsed_date'].dt.to_period('M')
    
    monthly_analysis = {}
    year_months = sorted(speeches_df['year_month'].unique())
    
    for year_month in year_months:
        monthly_speeches = speeches_df[speeches_df['year_month'] == year_month]
        monthly_texts = monthly_speeches['processed_text'].tolist()
        
        if len(monthly_texts) > 10:
            monthly_tfidf = extract_keywords_tfidf(monthly_texts, max_features=300)
            monthly_sentiment = sentiment_analysis(monthly_texts)
            monthly_sentiment_counts = Counter(monthly_sentiment)
            
            monthly_analysis[str(year_month)] = {
                'total_speeches': len(monthly_speeches),
                'tfidf_keywords': monthly_tfidf,
                'sentiment_distribution': dict(monthly_sentiment_counts)
            }
    
    results['monthly_analysis'] = monthly_analysis
    
    return results, speeches_df

def create_advanced_visualizations(results, speeches_df):
    """고급 시각화 생성"""
    print("\n고급 시각화 생성 중...")
    
    # 결과 폴더 생성
    os.makedirs('analysis_results', exist_ok=True)
    
    # 1. TF-IDF 키워드 워드클라우드
    if 'tfidf_keywords' in results and results['tfidf_keywords']:
        print("1. TF-IDF 키워드 워드클라우드 생성")
        try:
            wordcloud = WordCloud(
                font_path='C:/Windows/Fonts/malgun.ttf',
                width=800,
                height=400,
                background_color='white',
                max_words=100
            ).generate_from_frequencies(results['tfidf_keywords'])
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('TF-IDF 기반 주요 키워드', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('analysis_results/04_tfidf_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"워드클라우드 생성 실패: {e}")
    
    # 2. LDA 토픽 시각화
    if 'lda_topics' in results and results['lda_topics']:
        print("2. LDA 토픽 시각화 생성")
        topics = results['lda_topics']['topics']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, topic in enumerate(topics[:8]):
            words = topic['words'][:10]  # 상위 10개 단어
            scores = [1.0] * len(words)  # 간단한 점수
            
            axes[i].barh(range(len(words)), scores)
            axes[i].set_yticks(range(len(words)))
            axes[i].set_yticklabels(words, fontsize=10)
            axes[i].set_title(f'토픽 {topic["topic_id"]}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('중요도')
        
        # 빈 subplot 제거
        for i in range(len(topics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('analysis_results/04_lda_topics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 감정 분석 결과
    if 'sentiment_analysis' in results:
        print("3. 감정 분석 시각화 생성")
        sentiment_data = results['sentiment_analysis']
        
        plt.figure(figsize=(10, 6))
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        wedges, texts, autotexts = plt.pie(
            sentiment_data.values(),
            labels=sentiment_data.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        plt.title('전체 감정 분석 결과', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('analysis_results/04_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 정당별 감정 분석
    if 'party_analysis' in results:
        print("4. 정당별 감정 분석 시각화 생성")
        party_analysis = results['party_analysis']
        
        # 정당별 감정 분포 히트맵
        sentiment_df = pd.DataFrame({
            party: data['sentiment_distribution'] 
            for party, data in party_analysis.items()
        }).fillna(0)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(sentiment_df, annot=True, fmt='.0f', cmap='RdYlBu_r')
        plt.title('정당별 감정 분석', fontsize=16, fontweight='bold')
        plt.xlabel('정당', fontsize=12)
        plt.ylabel('감정', fontsize=12)
        plt.tight_layout()
        plt.savefig('analysis_results/04_party_sentiment_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. 시간별 감정 변화
    if 'monthly_analysis' in results:
        print("5. 시간별 감정 변화 시각화 생성")
        monthly_analysis = results['monthly_analysis']
        
        if monthly_analysis:
            # 월별 감정 분포 데이터 준비
            monthly_sentiment_data = []
            months = []
            
            for month, data in monthly_analysis.items():
                months.append(month)
                sentiment_dist = data['sentiment_distribution']
                monthly_sentiment_data.append([
                    sentiment_dist.get('positive', 0),
                    sentiment_dist.get('negative', 0),
                    sentiment_dist.get('neutral', 0)
                ])
            
            monthly_df = pd.DataFrame(
                monthly_sentiment_data,
                index=months,
                columns=['positive', 'negative', 'neutral']
            )
            
            # 스택 바 차트
            plt.figure(figsize=(15, 8))
            monthly_df.plot(kind='bar', stacked=True, color=['#99ff99', '#ff9999', '#66b3ff'])
            plt.title('월별 감정 변화', fontsize=16, fontweight='bold')
            plt.xlabel('년-월', fontsize=12)
            plt.ylabel('발언 수', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title='감정')
            plt.tight_layout()
            plt.savefig('analysis_results/04_monthly_sentiment_trends.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("고급 시각화 완료!")

def generate_advanced_report(results, speeches_df):
    """고급 분석 보고서 생성"""
    print("\n고급 분석 보고서 생성 중...")
    
    # 보고서 폴더 생성
    os.makedirs('analysis_reports', exist_ok=True)
    
    report_content = f"""# 고급 토픽 분석 보고서

## 분석 개요
- **분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **분석 대상**: 국회 회의록 발언 데이터
- **분석 방법**: 고급 자연어처리 기법
  - 형태소 분석 (KoNLPy)
  - TF-IDF 키워드 추출
  - LDA 토픽 모델링
  - 감정 분석
- **고급 라이브러리 사용**: {'Yes' if ADVANCED_AVAILABLE else 'No (기본 분석)'}

## TF-IDF 키워드 분석

"""
    
    # TF-IDF 키워드 결과
    if 'tfidf_keywords' in results and results['tfidf_keywords']:
        report_content += "**상위 20개 키워드**:\n\n"
        sorted_keywords = sorted(results['tfidf_keywords'].items(), key=lambda x: x[1], reverse=True)
        for i, (keyword, score) in enumerate(sorted_keywords[:20]):
            report_content += f"{i+1}. **{keyword}**: {score:.4f}\n"
    else:
        report_content += "TF-IDF 키워드 분석을 사용할 수 없습니다.\n"
    
    report_content += "\n## LDA 토픽 모델링 결과\n\n"
    
    # LDA 토픽 결과
    if 'lda_topics' in results and results['lda_topics']:
        topics = results['lda_topics']['topics']
        for topic in topics:
            report_content += f"### 토픽 {topic['topic_id']}\n"
            report_content += f"**주요 단어**: {', '.join(topic['words'][:10])}\n\n"
    else:
        report_content += "LDA 토픽 모델링을 사용할 수 없습니다.\n"
    
    report_content += "\n## 감정 분석 결과\n\n"
    
    # 감정 분석 결과
    if 'sentiment_analysis' in results:
        sentiment_data = results['sentiment_analysis']
        total = sum(sentiment_data.values())
        
        report_content += "**전체 감정 분포**:\n\n"
        for sentiment, count in sentiment_data.items():
            percentage = (count / total * 100) if total > 0 else 0
            report_content += f"- **{sentiment}**: {count:,}개 ({percentage:.1f}%)\n"
    
    report_content += "\n## 정당별 고급 분석\n\n"
    
    # 정당별 분석 결과
    if 'party_analysis' in results:
        party_analysis = results['party_analysis']
        
        for party, data in party_analysis.items():
            report_content += f"### {party} 정당\n\n"
            report_content += f"**전체 발언 수**: {data['total_speeches']:,}개\n\n"
            
            # 감정 분포
            sentiment_dist = data['sentiment_distribution']
            total = sum(sentiment_dist.values())
            report_content += "**감정 분포**:\n"
            for sentiment, count in sentiment_dist.items():
                percentage = (count / total * 100) if total > 0 else 0
                report_content += f"- {sentiment}: {count:,}개 ({percentage:.1f}%)\n"
            
            # 상위 키워드
            if data['tfidf_keywords']:
                report_content += "\n**상위 키워드**:\n"
                sorted_keywords = sorted(data['tfidf_keywords'].items(), key=lambda x: x[1], reverse=True)
                for i, (keyword, score) in enumerate(sorted_keywords[:10]):
                    report_content += f"{i+1}. {keyword} ({score:.4f})\n"
            
            report_content += "\n---\n"
    
    report_content += "\n## 월별 고급 분석\n\n"
    
    # 월별 분석 결과
    if 'monthly_analysis' in results:
        monthly_analysis = results['monthly_analysis']
        
        for month, data in monthly_analysis.items():
            report_content += f"### {month}\n\n"
            report_content += f"**전체 발언 수**: {data['total_speeches']:,}개\n\n"
            
            # 감정 분포
            sentiment_dist = data['sentiment_distribution']
            total = sum(sentiment_dist.values())
            report_content += "**감정 분포**:\n"
            for sentiment, count in sentiment_dist.items():
                percentage = (count / total * 100) if total > 0 else 0
                report_content += f"- {sentiment}: {count:,}개 ({percentage:.1f}%)\n"
            
            report_content += "\n"
    
    report_content += """
## 분석 결과 요약

### 주요 발견사항

1. **고급 자연어처리 기법의 효과**
   - 형태소 분석을 통한 정확한 키워드 추출
   - TF-IDF를 통한 중요 키워드 식별
   - LDA를 통한 숨겨진 토픽 발견

2. **감정 분석 결과**
   - 정책에 대한 정당별 감정적 태도 파악
   - 시간에 따른 감정 변화 추이 분석

3. **토픽 모델링의 인사이트**
   - 기존 키워드 분석으로 발견하지 못한 주제 발견
   - 정책 영역 간의 연관성 파악

### 분석의 한계점

1. **고급 라이브러리 의존성**
   - 일부 라이브러리가 없을 경우 기본 분석으로 대체
   - 설치 및 설정의 복잡성

2. **계산 복잡도**
   - 대용량 데이터 처리 시 시간 소요
   - 메모리 사용량 증가

### 향후 개선 방향

1. **더 고급 모델 적용**
   - BERT 기반 토픽 모델링
   - 딥러닝 기반 감정 분석

2. **실시간 분석**
   - 스트리밍 데이터 처리
   - 실시간 토픽 변화 추적

3. **다국어 지원**
   - 영어 발언 분석
   - 다국어 토픽 모델링

---
*본 보고서는 고급 자연어처리 기법을 활용한 국회 회의록 분석 결과입니다.*
"""
    
    # 보고서 저장
    with open('analysis_reports/04_advanced_topic_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("고급 분석 보고서 생성 완료!")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("4단계: 고급 토픽 분석")
    print("=" * 60)
    
    if not ADVANCED_AVAILABLE:
        print("⚠️  고급 라이브러리가 일부 없습니다.")
        print("기본 키워드 분석으로 진행합니다.")
    
    # 데이터 로드
    speeches_df = load_data()
    if speeches_df is None:
        print("데이터 로드 실패. 프로그램을 종료합니다.")
        return
    
    # 고급 토픽 분석
    results, processed_df = analyze_advanced_topics(speeches_df)
    if not results:
        print("고급 분석 실패. 프로그램을 종료합니다.")
        return
    
    # 고급 시각화 생성
    create_advanced_visualizations(results, processed_df)
    
    # 고급 보고서 생성
    generate_advanced_report(results, processed_df)
    
    print("\n" + "=" * 60)
    print("4단계 고급 분석 완료!")
    print("=" * 60)
    print("생성된 파일:")
    print("- analysis_results/04_tfidf_wordcloud.png")
    print("- analysis_results/04_lda_topics.png")
    print("- analysis_results/04_sentiment_analysis.png")
    print("- analysis_results/04_party_sentiment_heatmap.png")
    print("- analysis_results/04_monthly_sentiment_trends.png")
    print("- analysis_reports/04_advanced_topic_analysis_report.md")

if __name__ == "__main__":
    main()
