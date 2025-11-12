#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3단계: 시간별 정책 관심도 변화 분석
- 회기별 정책 토픽 트렌드 분석
- 월별 정책 관심도 변화 분석
- 정당별 시간별 정책 관심도 변화 분석
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

def parse_date(date_str):
    """날짜 문자열 파싱"""
    if pd.isna(date_str):
        return None
    
    try:
        # "2024년 6월 13일(목)" 형식 파싱
        date_str = str(date_str)
        match = re.search(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', date_str)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            return pd.Timestamp(year=year, month=month, day=day)
    except:
        pass
    
    return None

def get_policy_areas():
    """정책 영역 정의 (02와 동일)"""
    return {
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

def analyze_topic_by_time(speeches_df):
    """시간별 정책 토픽 분석"""
    print("\n시간별 정책 토픽 분석 시작...")
    
    # 날짜 파싱
    speeches_df['parsed_date'] = speeches_df['date'].apply(parse_date)
    speeches_df = speeches_df[speeches_df['parsed_date'].notna()].copy()
    
    # 시간 단위 추가
    speeches_df['year'] = speeches_df['parsed_date'].dt.year
    speeches_df['month'] = speeches_df['parsed_date'].dt.month
    speeches_df['year_month'] = speeches_df['year'].astype(str) + '-' + speeches_df['month'].astype(str).str.zfill(2)
    speeches_df['session'] = speeches_df['session'].astype(str)
    
    # 정책 영역 정의
    policy_areas = get_policy_areas()
    
    # 회기별 분석
    session_analysis = {}
    sessions = sorted(speeches_df['session'].unique())
    
    for session in sessions:
        session_speeches = speeches_df[speeches_df['session'] == session]
        if len(session_speeches) == 0:
            continue
        
        topic_scores = {}
        total_speeches = len(session_speeches)
        
        for topic, config in policy_areas.items():
            keywords = config['keywords']
            weight = config['weight']
            
            topic_count = 0
            for _, speech in session_speeches.iterrows():
                speech_text = str(speech.get('speech_content', '')).lower()
                if pd.isna(speech_text):
                    continue
                
                for keyword in keywords:
                    if keyword.lower() in speech_text:
                        topic_count += 1
            
            topic_score = (topic_count * weight) / total_speeches if total_speeches > 0 else 0
            topic_scores[topic] = {
                'score': topic_score,
                'count': topic_count,
                'percentage': (topic_count / total_speeches * 100) if total_speeches > 0 else 0
            }
        
        session_analysis[session] = {
            'total_speeches': total_speeches,
            'date_range': f"{session_speeches['parsed_date'].min():%Y-%m-%d} ~ {session_speeches['parsed_date'].max():%Y-%m-%d}",
            'topic_scores': topic_scores
        }
    
    # 월별 분석
    monthly_analysis = {}
    year_months = sorted(speeches_df['year_month'].unique())
    
    for year_month in year_months:
        monthly_speeches = speeches_df[speeches_df['year_month'] == year_month]
        if len(monthly_speeches) == 0:
            continue
        
        topic_scores = {}
        total_speeches = len(monthly_speeches)
        
        for topic, config in policy_areas.items():
            keywords = config['keywords']
            weight = config['weight']
            
            topic_count = 0
            for _, speech in monthly_speeches.iterrows():
                speech_text = str(speech.get('speech_content', '')).lower()
                if pd.isna(speech_text):
                    continue
                
                for keyword in keywords:
                    if keyword.lower() in speech_text:
                        topic_count += 1
            
            topic_score = (topic_count * weight) / total_speeches if total_speeches > 0 else 0
            topic_scores[topic] = {
                'score': topic_score,
                'count': topic_count,
                'percentage': (topic_count / total_speeches * 100) if total_speeches > 0 else 0
            }
        
        monthly_analysis[year_month] = {
            'total_speeches': total_speeches,
            'topic_scores': topic_scores
        }
    
    # 정당별 시간별 분석
    party_time_analysis = {}
    parties = speeches_df['speaker_party'].dropna().unique()
    
    for party in parties:
        if pd.isna(party) or party == '':
            continue
        
        party_speeches = speeches_df[speeches_df['speaker_party'] == party]
        
        # 정당별 월별 분석
        party_monthly_analysis = {}
        
        for year_month in year_months:
            monthly_speeches = party_speeches[party_speeches['year_month'] == year_month]
            if len(monthly_speeches) == 0:
                continue
            
            topic_scores = {}
            total_speeches = len(monthly_speeches)
            
            for topic, config in policy_areas.items():
                keywords = config['keywords']
                weight = config['weight']
                
                topic_count = 0
                for _, speech in monthly_speeches.iterrows():
                    speech_text = str(speech.get('speech_content', '')).lower()
                    if pd.isna(speech_text):
                        continue
                    
                    for keyword in keywords:
                        if keyword.lower() in speech_text:
                            topic_count += 1
                
                topic_score = (topic_count * weight) / total_speeches if total_speeches > 0 else 0
                topic_scores[topic] = {
                    'score': topic_score,
                    'count': topic_count
                }
            
            party_monthly_analysis[year_month] = {
                'total_speeches': total_speeches,
                'topic_scores': topic_scores
            }
        
        party_time_analysis[party] = {
            'monthly_analysis': party_monthly_analysis
        }
    
    return {
        'session_analysis': session_analysis,
        'monthly_analysis': monthly_analysis,
        'party_time_analysis': party_time_analysis
    }

def create_time_visualizations(time_analysis):
    """시간별 변화 시각화 생성"""
    print("\n시간별 변화 시각화 생성 중...")
    
    # 결과 폴더 생성
    os.makedirs('analysis_results', exist_ok=True)
    
    session_analysis = time_analysis['session_analysis']
    monthly_analysis = time_analysis['monthly_analysis']
    party_time_analysis = time_analysis['party_time_analysis']
    
    policy_areas = get_policy_areas()
    topics = list(policy_areas.keys())
    
    # 1. 회기별 정책 토픽 트렌드
    sessions = sorted(session_analysis.keys())
    if sessions:
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        axes = axes.flatten()
        
        for i, topic in enumerate(topics[:4]):  # 상위 4개 토픽만 표시
            session_scores = []
            session_labels = []
            
            for session in sessions:
                score = session_analysis[session]['topic_scores'][topic]['score']
                session_scores.append(score)
                session_labels.append(f"제{session}회")
            
            axes[i].plot(range(len(session_labels)), session_scores, marker='o', linewidth=2, markersize=8)
            axes[i].set_title(f'{topic} 정책 영역 - 회기별 변화', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('회기', fontsize=10)
            axes[i].set_ylabel('토픽 점수', fontsize=10)
            axes[i].set_xticks(range(len(session_labels)))
            axes[i].set_xticklabels(session_labels, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
        
        # 빈 subplot 제거
        for i in range(len(topics[:4]), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('analysis_results/03_session_topic_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 월별 정책 토픽 트렌드 (전체)
    year_months = sorted(monthly_analysis.keys())
    if year_months:
        fig, ax = plt.subplots(figsize=(18, 10))
        
        for topic in topics:
            monthly_scores = []
            for year_month in year_months:
                score = monthly_analysis[year_month]['topic_scores'][topic]['score']
                monthly_scores.append(score)
            
            ax.plot(year_months, monthly_scores, marker='o', label=topic, linewidth=2, markersize=6)
        
        ax.set_title('월별 정책 토픽 관심도 변화', fontsize=16, fontweight='bold')
        ax.set_xlabel('년-월', fontsize=12)
        ax.set_ylabel('토픽 점수', fontsize=12)
        ax.set_xticks(year_months[::2])  # 너무 많은 레이블 방지
        ax.set_xticklabels(year_months[::2], rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis_results/03_monthly_topic_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 정당별 월별 주요 토픽 변화
    parties = list(party_time_analysis.keys())
    # 미분류 정당 제외
    parties = [party for party in parties if party != '미분류']
    
    if parties:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, party in enumerate(parties[:4]):  # 최대 4개 정당 (2x2)
            if i >= len(axes):
                break
            
            party_monthly = party_time_analysis[party]['monthly_analysis']
            
            # 상위 3개 토픽 찾기
            avg_scores = {}
            for topic in topics:
                scores = []
                for year_month in year_months:
                    if year_month in party_monthly:
                        score = party_monthly[year_month]['topic_scores'][topic]['score']
                        scores.append(score)
                if scores:
                    avg_scores[topic] = np.mean(scores)
            
            top_topics = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_topic_names = [item[0] for item in top_topics]
            
            for topic in top_topic_names:
                monthly_scores = []
                for year_month in year_months:
                    if year_month in party_monthly:
                        score = party_monthly[year_month]['topic_scores'][topic]['score']
                        monthly_scores.append(score)
                    else:
                        monthly_scores.append(0)
                
                axes[i].plot(year_months, monthly_scores, marker='o', label=topic, linewidth=2, markersize=5)
            
            axes[i].set_title(f'{party} 정당 - 월별 주요 토픽 변화', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('년-월', fontsize=10)
            axes[i].set_ylabel('토픽 점수', fontsize=10)
            axes[i].set_xticks(year_months[::2])
            axes[i].set_xticklabels(year_months[::2], rotation=45, ha='right', fontsize=9)
            axes[i].legend(fontsize=9)
            axes[i].grid(True, alpha=0.3)
        
        # 빈 subplot 제거
        for i in range(len(parties), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('analysis_results/03_party_monthly_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("시간별 변화 시각화 완료!")

def generate_time_report(time_analysis):
    """시간별 변화 분석 보고서 생성"""
    print("\n시간별 변화 분석 보고서 생성 중...")
    
    # 보고서 폴더 생성
    os.makedirs('analysis_reports', exist_ok=True)
    
    session_analysis = time_analysis['session_analysis']
    monthly_analysis = time_analysis['monthly_analysis']
    party_time_analysis = time_analysis['party_time_analysis']
    
    policy_areas = get_policy_areas()
    topics = list(policy_areas.keys())
    
    report_content = f"""# 시간별 정책 관심도 변화 분석 보고서

## 분석 개요
- **분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **분석 대상**: 국회 회의록 발언 데이터
- **분석 방법**: 시간별 키워드 기반 토픽 분석
- **정책 영역**: 10개 주요 정책 영역

## 회기별 정책 토픽 분석

"""
    
    sessions = sorted(session_analysis.keys())
    for session in sessions:
        data = session_analysis[session]
        report_content += f"""
### 제{session}회 국회
- **기간**: {data['date_range']}
- **전체 발언 수**: {data['total_speeches']:,}건

**정책 영역별 관심도 (상위 5개)**:
"""
        
        topic_scores = data['topic_scores']
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
        
        report_content += "\n---\n"
    
    # 월별 트렌드 요약
    report_content += """
## 월별 정책 토픽 트렌드 요약

### 주요 정책 영역별 변화 추이

"""
    
    year_months = sorted(monthly_analysis.keys())
    for topic in topics:
        monthly_scores = []
        for year_month in year_months:
            score = monthly_analysis[year_month]['topic_scores'][topic]['score']
            monthly_scores.append((year_month, score))
        
        if monthly_scores:
            max_month = max(monthly_scores, key=lambda x: x[1])
            min_month = min(monthly_scores, key=lambda x: x[1])
            
            report_content += f"""
**{topic}**
- 최고 관심도: {max_month[0]} ({max_month[1]:.4f})
- 최저 관심도: {min_month[0]} ({min_month[1]:.4f})
- 평균 관심도: {np.mean([s[1] for s in monthly_scores]):.4f}
"""
    
    # 정당별 시간별 분석
    report_content += """
## 정당별 시간별 정책 관심도 변화

"""
    
    parties = list(party_time_analysis.keys())
    for party in parties:
        report_content += f"""
### {party} 정당

**월별 주요 정책 관심도 변화**:
"""
        
        party_monthly = party_time_analysis[party]['monthly_analysis']
        
        # 평균 점수로 상위 토픽 찾기
        avg_scores = {}
        for topic in topics:
            scores = []
            for year_month in year_months:
                if year_month in party_monthly:
                    score = party_monthly[year_month]['topic_scores'][topic]['score']
                    scores.append(score)
            if scores:
                avg_scores[topic] = np.mean(scores)
        
        sorted_topics = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for i, (topic, avg_score) in enumerate(sorted_topics):
            report_content += f"""
{i+1}. **{topic}**: 평균 점수 {avg_score:.4f}
"""
        
        report_content += "\n---\n"
    
    # 분석 결과 요약
    report_content += """
## 분석 결과 요약

### 주요 발견사항

1. **시간에 따른 정책 관심도 변화**
   - 특정 시기에 정책 관심도가 집중되는 경향 관찰
   - 회기별, 월별로 정책 우선순위가 변화함

2. **정책 영역별 트렌드**
   - 일부 정책 영역은 시간에 따라 일정한 관심도 유지
   - 특정 이벤트 발생 시 해당 정책 영역의 관심도 급증

3. **정당별 정책 관심도 변화 패턴**
   - 정당별로 주요 관심 정책 영역의 시간적 변화 패턴이 다름
   - 정당의 정책 우선순위가 시간에 따라 조정됨

### 분석의 한계점

1. **시간 범위의 한계**
   - 분석 기간이 제한적일 수 있음
   - 장기 트렌드 파악을 위해서는 더 긴 기간의 데이터 필요

2. **이벤트 연관성 분석의 부재**
   - 특정 이벤트와 정책 관심도 변화의 인과관계 분석 필요
   - 맥락적 정보가 부족할 수 있음

### 향후 개선 방향

1. **이벤트 기반 분석**
   - 특정 사고, 선거, 법안 통과 등과 정책 관심도 변화의 상관관계 분석

2. **예측 모델 개발**
   - 과거 데이터를 기반으로 향후 정책 관심도 예측

3. **계절성 분석**
   - 분기별, 계절별 정책 관심도 패턴 분석

---
*본 보고서는 국회 회의록 데이터를 기반으로 한 시간별 정책 관심도 변화 분석 결과입니다.*
"""
    
    # 보고서 저장
    with open('analysis_reports/03_time_based_topic_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("시간별 변화 분석 보고서 생성 완료!")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("3단계: 시간별 정책 관심도 변화 분석")
    print("=" * 60)
    
    # 데이터 로드
    speeches_df = load_data()
    if speeches_df is None:
        print("데이터 로드 실패. 프로그램을 종료합니다.")
        return
    
    # 시간별 정책 토픽 분석
    time_analysis = analyze_topic_by_time(speeches_df)
    if not time_analysis:
        print("시간별 분석 실패. 프로그램을 종료합니다.")
        return
    
    # 시각화 생성
    create_time_visualizations(time_analysis)
    
    # 보고서 생성
    generate_time_report(time_analysis)
    
    print("\n" + "=" * 60)
    print("3단계 분석 완료!")
    print("=" * 60)
    print("생성된 파일:")
    print("- analysis_results/03_session_topic_trends.png")
    print("- analysis_results/03_monthly_topic_trends.png")
    print("- analysis_results/03_party_monthly_trends.png")
    print("- analysis_reports/03_time_based_topic_analysis_report.md")

if __name__ == "__main__":
    main()
