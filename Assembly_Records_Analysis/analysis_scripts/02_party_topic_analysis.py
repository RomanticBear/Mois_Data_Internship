#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2단계: 정당별 토픽 분석
- 각 정당이 어떤 정책 영역에 더 관심을 가지고 있는지 분석
- 정당별 토픽 분포 및 특성 파악
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
    all_headers = []
    
    # data/with_party 폴더의 모든 회의록 데이터 수집
    minutes_dir = "data/with_party"
    if not os.path.exists(minutes_dir):
        print(f"경고: {minutes_dir} 폴더를 찾을 수 없습니다.")
        return None, None
    
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
                    if 'party' in df.columns:  # 컬럼명이 'party'임
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
        return None, None
    
    # 데이터 통합
    speeches_df = pd.concat(all_speeches, ignore_index=True)
    print(f"총 {len(speeches_df)}개의 발언 데이터 로드 완료")
    
    return speeches_df, None

def analyze_party_topics(speeches_df):
    """정당별 토픽 분석"""
    print("\n정당별 토픽 분석 시작...")
    
    # 정책 영역 정의 (1단계와 동일)
    policy_areas = {
        '안전관리': {
            'keywords': ['안전', '재난', '소방', '경찰', '사고', '참사', '화재', '안전관리', '재난안전',
                        '소방청', '경찰청', '안전사고', '재해', '방재', '구조', '응급', '119', '112',
                        '행정안전부', '재난관리', '안전정책', '재난대응', '소방정책', '경찰정책'],
            'weight': 1.0
        },
        '경제정책': {
            'keywords': ['경제', '성장', '투자', '고용', '일자리', '기업', '중소기업', '창업', '혁신',
                        '산업', '제조업', '서비스업', '수출', '수입', '무역', '경제정책', '성장정책',
                        '기획재정부', '산업통상자원부', '중소벤처기업부', '고용노동부', '경제부총리'],
            'weight': 1.0
        },
        '사회복지': {
            'keywords': ['복지', '사회보장', '연금', '보험', '의료', '건강', '보건', '사회복지', '기초생활',
                        '생활보호', '장애인', '노인', '아동', '청소년', '여성', '가족', '보육', '양육',
                        '보건복지부', '여성가족부', '복지정책', '사회보장', '의료보험', '국민연금'],
            'weight': 1.0
        },
        '교육정책': {
            'keywords': ['교육', '학교', '학생', '교사', '대학', '고등학교', '중학교', '초등학교', '유치원',
                        '교육부', '교육정책', '교육과정', '입시', '대입', '수능', '학원', '사교육',
                        '교육개혁', '교육평등', '교육기회', '교육비', '장학금', '교육복지'],
            'weight': 1.0
        },
        '환경정책': {
            'keywords': ['환경', '기후', '탄소', '에너지', '재생에너지', '태양광', '풍력', '원자력', '석탄',
                        '환경부', '환경정책', '기후변화', '탄소중립', '녹색성장', '대기', '수질', '토양',
                        '환경보호', '생태', '자연', '환경오염', '미세먼지', '온실가스'],
            'weight': 1.0
        },
        '국방외교': {
            'keywords': ['국방', '군사', '군인', '군대', '국방부', '국방정책', '안보', '보안', '외교',
                        '외교부', '외교정책', '국제', '국제관계', '외교관계', '외교관', '대사', '영사',
                        '국제협력', '국제기구', 'UN', '국제법', '외교안보', '국방외교'],
            'weight': 1.0
        },
        '지방분권': {
            'keywords': ['지방', '지역', '지자체', '시도', '시군구', '지방자치', '지방분권', '지역균형',
                        '행정안전부', '지방정부', '지역발전', '지역경제', '지역정책', '지역균형발전',
                        '지방재정', '지역재정', '지역주민', '지역사회', '지역발전정책'],
            'weight': 1.0
        },
        '인사관리': {
            'keywords': ['인사', '공무원', '인사관리', '인사정책', '공무원채용', '공무원시험', '공무원연수',
                        '인사혁신처', '인사혁신', '공무원제도', '공무원복무', '공무원연금', '공무원보수',
                        '공무원교육', '공무원훈련', '공무원평가', '공무원승진', '공무원징계'],
            'weight': 1.0
        },
        '법무사법': {
            'keywords': ['법무', '사법', '법원', '검찰', '법무부', '법무정책', '사법정책', '법률', '법령',
                        '입법', '법제처', '법제', '법률안', '법률제정', '법률개정', '법률폐지', '법률해석',
                        '사법제도', '사법개혁', '사법정책', '법무정책'],
            'weight': 1.0
        },
        '과학기술': {
            'keywords': ['과학', '기술', '연구', '개발', 'R&D', '과학기술', '과학기술정보통신부', '과학기술정책',
                        '연구개발', '기술혁신', '과학기술혁신', '과학기술정책', '과학기술연구', '과학기술개발',
                        '과학기술인', '과학기술자', '과학기술자', '과학기술연구원', '과학기술연구소'],
            'weight': 1.0
        }
    }
    
    # 정당별 토픽 분석
    party_topic_analysis = {}
    
    # 정당별로 데이터 분리
    parties = speeches_df['speaker_party'].dropna().unique()
    print(f"발견된 정당: {list(parties)}")
    
    for party in parties:
        if pd.isna(party) or party == '':
            continue
            
        print(f"\n{party} 정당 분석 중...")
        
        # 해당 정당의 발언만 추출
        party_speeches = speeches_df[speeches_df['speaker_party'] == party]
        
        if len(party_speeches) == 0:
            continue
        
        # 정책 영역별 키워드 매칭
        topic_scores = {}
        total_speeches = len(party_speeches)
        
        for topic, config in policy_areas.items():
            keywords = config['keywords']
            weight = config['weight']
            
            # 해당 정당의 발언에서 키워드 매칭
            topic_count = 0
            topic_speeches = []
            
            for _, speech in party_speeches.iterrows():
                speech_text = str(speech.get('speech_content', ''))
                if pd.isna(speech_text):
                    continue
                
                speech_text_lower = speech_text.lower()
                matched_keywords = []
                
                for keyword in keywords:
                    if keyword.lower() in speech_text_lower:
                        matched_keywords.append(keyword)
                        topic_count += 1
                
                if matched_keywords:
                    topic_speeches.append({
                        'speech_id': speech.get('speech_id', ''),
                        'speaker_name': speech.get('speaker_name', ''),
                        'matched_keywords': matched_keywords,
                        'speech_content': speech_text[:100] + '...' if len(speech_text) > 100 else speech_text
                    })
            
            # 토픽 점수 계산
            topic_score = (topic_count * weight) / total_speeches if total_speeches > 0 else 0
            
            topic_scores[topic] = {
                'score': topic_score,
                'count': topic_count,
                'speeches': topic_speeches,
                'percentage': (len(topic_speeches) / total_speeches * 100) if total_speeches > 0 else 0
            }
        
        party_topic_analysis[party] = {
            'total_speeches': total_speeches,
            'topic_scores': topic_scores
        }
    
    return party_topic_analysis

def create_party_topic_visualizations(party_topic_analysis):
    """정당별 토픽 시각화"""
    print("\n정당별 토픽 시각화 생성 중...")
    
    # 결과 폴더 생성
    os.makedirs('analysis_results', exist_ok=True)
    
    # 1. 정당별 토픽 점수 히트맵
    parties = list(party_topic_analysis.keys())
    topics = list(party_topic_analysis[parties[0]]['topic_scores'].keys()) if parties else []
    
    if not parties or not topics:
        print("시각화할 데이터가 없습니다.")
        return
    
    # 히트맵 데이터 준비
    heatmap_data = []
    for party in parties:
        party_scores = []
        for topic in topics:
            score = party_topic_analysis[party]['topic_scores'][topic]['score']
            party_scores.append(score)
        heatmap_data.append(party_scores)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=parties, columns=topics)
    
    # 히트맵 생성
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': '토픽 점수'})
    plt.title('정당별 정책 토픽 관심도', fontsize=16, fontweight='bold')
    plt.xlabel('정책 영역', fontsize=12)
    plt.ylabel('정당', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('analysis_results/02_party_topic_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 정당별 상위 3개 토픽 막대 그래프
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, party in enumerate(parties[:4]):  # 상위 4개 정당만 표시
        if i >= len(axes):
            break
            
        topic_scores = party_topic_analysis[party]['topic_scores']
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        topics_names = [item[0] for item in sorted_topics[:5]]  # 상위 5개 토픽
        scores = [item[1]['score'] for item in sorted_topics[:5]]
        
        axes[i].bar(range(len(topics_names)), scores, color='skyblue', alpha=0.7)
        axes[i].set_title(f'{party} 정당 - 상위 5개 정책 관심도', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('정책 영역', fontsize=10)
        axes[i].set_ylabel('토픽 점수', fontsize=10)
        axes[i].set_xticks(range(len(topics_names)))
        axes[i].set_xticklabels(topics_names, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3)
        
        # 값 표시
        for j, score in enumerate(scores):
            axes[i].text(j, score + 0.001, f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 빈 subplot 제거
    for i in range(len(parties), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('analysis_results/02_party_top_topics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 토픽별 정당 분포
    plt.figure(figsize=(14, 10))
    
    # 각 토픽별로 정당들의 점수 비교
    topic_party_scores = {}
    for topic in topics:
        topic_party_scores[topic] = []
        for party in parties:
            score = party_topic_analysis[party]['topic_scores'][topic]['score']
            topic_party_scores[topic].append(score)
    
    # 서브플롯 생성
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, topic in enumerate(topics):
        if i >= len(axes):
            break
            
        party_scores = topic_party_scores[topic]
        colors = plt.cm.Set3(np.linspace(0, 1, len(parties)))
        
        bars = axes[i].bar(parties, party_scores, color=colors, alpha=0.7)
        axes[i].set_title(f'{topic} 정책 영역', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('토픽 점수', fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # 최고 점수 정당 강조
        max_score = max(party_scores)
        max_idx = party_scores.index(max_score)
        bars[max_idx].set_color('red')
        bars[max_idx].set_alpha(0.8)
    
    # 빈 subplot 제거
    for i in range(len(topics), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('analysis_results/02_topic_party_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("정당별 토픽 시각화 완료!")

def generate_party_topic_report(party_topic_analysis):
    """정당별 토픽 분석 보고서 생성"""
    print("\n정당별 토픽 분석 보고서 생성 중...")
    
    # 보고서 폴더 생성
    os.makedirs('analysis_reports', exist_ok=True)
    
    report_content = f"""# 정당별 토픽 분석 보고서

## 분석 개요
- **분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **분석 대상**: 국회 회의록 발언 데이터
- **분석 방법**: 키워드 기반 토픽 분석
- **정책 영역**: 10개 주요 정책 영역

## 정당별 분석 결과

"""
    
    # 각 정당별 상세 분석
    for party, data in party_topic_analysis.items():
        total_speeches = data['total_speeches']
        topic_scores = data['topic_scores']
        
        report_content += f"""
### {party} 정당

**전체 발언 수**: {total_speeches:,}건

**정책 영역별 관심도 (상위 5개)**:
"""
        
        # 상위 5개 토픽 정렬
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        for i, (topic, info) in enumerate(sorted_topics[:5]):
            score = info['score']
            count = info['count']
            percentage = info['percentage']
            
            report_content += f"""
{i+1}. **{topic}**
   - 토픽 점수: {score:.4f}
   - 키워드 매칭 수: {count:,}회
   - 발언 비율: {percentage:.2f}%
"""
        
        # 해당 정당의 특징적인 발언 예시
        report_content += f"""
**주요 발언 예시**:
"""
        
        # 가장 관심이 높은 토픽의 발언 예시
        top_topic = sorted_topics[0][0]
        top_topic_speeches = sorted_topics[0][1]['speeches']
        
        if top_topic_speeches:
            report_content += f"""
- **{top_topic}** 관련 발언:
"""
            for i, speech in enumerate(top_topic_speeches[:3]):  # 상위 3개만
                report_content += f"""
  {i+1}. {speech['speaker_name']}: "{speech['speech_content']}"
     (매칭 키워드: {', '.join(speech['matched_keywords'])})
"""
        
        report_content += "\n---\n"
    
    # 정당별 비교 분석
    report_content += """
## 정당별 비교 분석

### 정당별 정책 관심도 순위

"""
    
    # 각 토픽별로 정당 순위 매기기
    for topic in list(party_topic_analysis[list(party_topic_analysis.keys())[0]]['topic_scores'].keys()):
        report_content += f"""
#### {topic} 정책 영역
"""
        
        # 해당 토픽에서 정당별 점수 정렬
        party_scores = []
        for party, data in party_topic_analysis.items():
            score = data['topic_scores'][topic]['score']
            party_scores.append((party, score))
        
        party_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (party, score) in enumerate(party_scores):
            report_content += f"{i+1}. {party}: {score:.4f}\n"
        
        report_content += "\n"
    
    # 분석 결과 요약
    report_content += """
## 분석 결과 요약

### 주요 발견사항

1. **정당별 정책 관심도 차이**
   - 각 정당마다 특정 정책 영역에 대한 관심도가 다름
   - 정당의 이념적 성향과 정책 우선순위가 발언 패턴에 반영됨

2. **정책 영역별 정당 분포**
   - 특정 정책 영역에서는 특정 정당이 압도적으로 높은 관심도 보임
   - 정당 간 정책 관심도 격차가 존재

3. **정당별 정책 특성**
   - 각 정당의 고유한 정책 아이덴티티 확인 가능
   - 정당별 정책 우선순위의 차이점 명확히 드러남

### 분석의 한계점

1. **키워드 기반 분석의 한계**
   - 단순 키워드 매칭으로는 문맥적 의미 파악 어려움
   - 동의어나 유사 표현의 누락 가능성

2. **정당 분류의 한계**
   - 정당명이 정확하지 않은 경우 분석에서 제외
   - 정당 소속이 불분명한 발언자들의 발언 제외

3. **시간적 변화 미고려**
   - 정당의 정책 관심도는 시간에 따라 변화할 수 있음
   - 시계열 분석을 통한 정책 관심도 변화 추적 필요

### 향후 개선 방향

1. **고도화된 토픽 분석**
   - LDA, BERT 등 고급 토픽 모델링 기법 적용
   - 문맥적 의미를 고려한 토픽 분석

2. **시계열 분석**
   - 정당별 정책 관심도의 시간적 변화 추적
   - 선거 주기와 정책 관심도 변화의 상관관계 분석

3. **정당별 정책 특성 심화 분석**
   - 정당별 정책 우선순위 변화 분석
   - 정당 간 정책 견해 차이의 정량적 측정

---
*본 보고서는 국회 회의록 데이터를 기반으로 한 정당별 토픽 분석 결과입니다.*
"""
    
    # 보고서 저장
    with open('analysis_reports/02_party_topic_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("정당별 토픽 분석 보고서 생성 완료!")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("2단계: 정당별 토픽 분석")
    print("=" * 60)
    
    # 데이터 로드
    speeches_df, headers_df = load_data()
    if speeches_df is None:
        print("데이터 로드 실패. 프로그램을 종료합니다.")
        return
    
    # 정당별 토픽 분석
    party_topic_analysis = analyze_party_topics(speeches_df)
    if not party_topic_analysis:
        print("정당별 토픽 분석 실패. 프로그램을 종료합니다.")
        return
    
    # 시각화 생성
    create_party_topic_visualizations(party_topic_analysis)
    
    # 보고서 생성
    generate_party_topic_report(party_topic_analysis)
    
    print("\n" + "=" * 60)
    print("2단계 분석 완료!")
    print("=" * 60)
    print("생성된 파일:")
    print("- analysis_results/02_party_topic_heatmap.png")
    print("- analysis_results/02_party_top_topics.png") 
    print("- analysis_results/02_topic_party_distribution.png")
    print("- analysis_reports/02_party_topic_analysis_report.md")

if __name__ == "__main__":
    main()
