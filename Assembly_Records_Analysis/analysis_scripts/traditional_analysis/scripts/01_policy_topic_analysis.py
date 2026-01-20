#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국회 정책 토픽 변화 추이 분석 - 목표에 집중한 최고의 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
from datetime import datetime
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_policy_data():
    """정책 관련 데이터만 로드"""
    print("📊 정책 관련 데이터 로딩 중...")
    
    all_data = []
    data_dir = '../../data'  # data/제oo회/ 구조로 변경
    sessions = [d for d in os.listdir(data_dir) if d.startswith('제') and os.path.isdir(os.path.join(data_dir, d))]
    sessions.sort()
    
    for session in sessions:
        session_dir = os.path.join(data_dir, session)
        speech_files = [f for f in os.listdir(session_dir) if 'speeches' in f]
        
        for file in speech_files:
            file_path = os.path.join(session_dir, file)
            df = pd.read_csv(file_path)
            df['session_name'] = session
            df['file_name'] = file
            all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ 총 {len(combined_df):,}개 발언 로드 완료")
    return combined_df

def filter_policy_speeches(df):
    """정책 관련 발언만 필터링"""
    print("\n 정책 관련 발언 필터링 중...")
    
    # 정책 관련 키워드
    policy_keywords = [
        '정책', '법안', '예산', '복지', '경제', '안전', '지방', '국민', '사회', '발전',
        '해결', '문제', '과제', '지원', '대책', '방안', '제도', '개선', '강화', '추진',
        '계획', '방향', '목표', '현안', '질의', '감사', '청문회', '인사', '임명',
        '재난', '소방', '경찰', '분권', '지역', '선거', '투표', '기금', '결산'
    ]
    
    # 의사진행 키워드 (제외할 키워드)
    procedural_keywords = [
        '의석을 정돈해', '성원이 되었으므로', '회의를 개최', '회의를 개회',
        '감사의 말씀', '고맙습니다', '다음은', '말씀해 주십시오', '이의 없으십니까',
        '의결하겠습니다', '상정합니다', '안건을 상정', '회의를 마치겠습니다',
        '좋습니다', '네', '예', '아니요', '그렇습니다', '맞습니다'
    ]
    
    procedural_excluded = 0
    short_speech_excluded = 0
    insufficient_policy_keywords = 0

    def is_policy_speech(text, speaker_position):
        nonlocal procedural_excluded, short_speech_excluded, insufficient_policy_keywords
        if pd.isna(text):
            return False
        text_str = str(text).lower()
        speaker_pos = str(speaker_position).lower()
        # 위원장의 의사진행 발언 제외
        if '위원장' in speaker_pos:
            if any(keyword in text_str for keyword in procedural_keywords):
                procedural_excluded += 1
                return False
        # 정책 관련 키워드 포함 여부
        policy_count = sum(1 for keyword in policy_keywords if keyword in text_str)
        # 발언 길이 고려 (너무 짧으면 제외)
        if len(text_str) < 50:
            short_speech_excluded += 1
            return False
        if policy_count < 2:
            insufficient_policy_keywords += 1
            return False
        return True
    policy_mask = df.apply(lambda row: is_policy_speech(row['speech_text'], row['speaker_position']), axis=1)
    policy_df = df[policy_mask].copy()
    print(f"전체 발언: {len(df):,}개")
    print(f"의사진행 키워드(프로시저)로 제외: {procedural_excluded:,}개")
    print(f"너무 짧아서 제외: {short_speech_excluded:,}개")
    print(f"정책 키워드 부족(<2개) 제외: {insufficient_policy_keywords:,}개")
    
    print(f"📊 정책 발언 필터링 결과:")
    print(f"  - 전체 발언: {len(df):,}개")
    print(f"  - 정책 발언: {len(policy_df):,}개 ({len(policy_df)/len(df)*100:.1f}%)")
    
    return policy_df

def extract_policy_topics(df):
    """정책 토픽 추출"""
    print("\n 정책 토픽 추출 중...")
    
    # 정책 영역별 키워드 정의
    policy_areas = {
        '안전관리': {
            'keywords': ['안전', '재난', '소방', '경찰', '사고', '참사', '화재', '안전관리', '재난안전',
                        '소방청', '경찰청', '안전사고', '재해', '방재', '구조', '응급', '119', '112',
                        '행정안전부', '재난관리', '안전정책', '재난대응', '소방정책', '경찰정책'],
            'weight': 1.0
        },
        '지방분권': {
            'keywords': ['지방', '분권', '지역', '지자체', '지방자치', '균형발전', '지방세', '지역균형',
                        '지방분권', '지역발전', '지방정부', '지역경제', '지방공기업', '지방교부세',
                        '인구감소지역', '접경지역', '지방자치법', '지역균형발전'],
            'weight': 1.0
        },
        '예산결산': {
            'keywords': ['예산', '결산', '회계', '기금', '재정', '예산안', '결산서', '회계연도',
                        '예산집행', '재정지원', '보조금', '지방교부세', '기금관리', '재정정책',
                        '예산편성', '예산심의', '예산집행'],
            'weight': 1.0
        },
        '인사관리': {
            'keywords': ['인사', '임명', '청문회', '후보자', '장관', '위원장', '인사혁신', '공무원',
                        '임용', '승진', '인사제도', '인사정책', '인사혁신처', '인사청문회',
                        '인사청문요청안', '인사혁신처장'],
            'weight': 1.0
        },
        '법안심사': {
            'keywords': ['법안', '법률', '개정', '제정', '심사', '통과', '법률안', '법안심사',
                        '입법', '법제', '법령', '법규', '특별조치법안', '특별법안', '일부개정법률안'],
            'weight': 1.0
        },
        '국정감사': {
            'keywords': ['감사', '현안', '질의', '조사', '청문회', '국정조사', '감사원', '감사위원',
                        '현안질의', '국정감사', '피해조사', '현안질의'],
            'weight': 1.0
        },
        '선거관리': {
            'keywords': ['선거', '투표', '선거관리', '선관위', '선거제도', '선거법', '공직선거',
                        '선거공영제', '선거비용', '중앙선거관리위원회', '국민투표법', '지방선거'],
            'weight': 1.0
        },
        '복지정책': {
            'keywords': ['복지', '복지정책', '복지제도', '복지혜택', '복지서비스', '사회보장',
                        '복지국가', '복지예산', '민생회복지원금', '민생회복지원법', '민생'],
            'weight': 1.0
        },
        '경제정책': {
            'keywords': ['경제', '경제정책', '경제성장', '경제발전', '경제지표', '경제지원',
                        '경제활성화', '경제회복', '경제활동', '경제적', '경제자유특구'],
            'weight': 1.0
        }
    }
    
    # 각 발언을 정책 영역별로 분류
    def classify_policy_area(text):
        if pd.isna(text):
            return '기타'
        
        text_str = str(text).lower()
        area_scores = {}
        
        for area, info in policy_areas.items():
            score = 0
            for keyword in info['keywords']:
                if keyword in text_str:
                    score += info['weight']
            area_scores[area] = score
        
        if max(area_scores.values()) > 0:
            return max(area_scores, key=area_scores.get)
        else:
            return '기타'
    
    df['policy_area'] = df['speech_text'].apply(classify_policy_area)
    
    # 정책 영역별 통계
    area_counts = df['policy_area'].value_counts()
    print(f"📊 정책 영역별 발언 수:")
    for area, count in area_counts.items():
        print(f"  - {area}: {count:,}개 ({count/len(df)*100:.1f}%)")
    
    return df, policy_areas

def analyze_temporal_trends(df):
    """영역별 발언 변화 추이 분석"""
    print("\n영역별 발언 변화 추이 분석 중...")
    
    # 회기별 정책 영역별 발언 수
    trend_data = df.groupby(['session_name', 'policy_area']).size().unstack(fill_value=0)
    
    # 회기 번호 추출 및 정렬
    def extract_session_number(session_name):
        return int(session_name.replace('제', '').replace('회', ''))
    
    trend_data['session_num'] = trend_data.index.map(extract_session_number)
    trend_data = trend_data.sort_values('session_num')
    
    # 정책 영역별 총 발언 수
    area_totals = trend_data.drop('session_num', axis=1).sum().sort_values(ascending=False)
    
    print(f"정책 영역별 총 발언 수:")
    for area, count in area_totals.items():
        print(f"  - {area}: {count:,}개")
    
    return trend_data, area_totals

def analyze_policy_keywords(df):
    """정책 키워드 분석"""
    print("\n정책 키워드 분석 중...")
    
    # 정책 영역별 키워드 추출
    area_keywords = defaultdict(list)
    
    for area in df['policy_area'].unique():
        if area == '기타':
            continue
        
        area_speeches = df[df['policy_area'] == area]['speech_text']
        
        # 키워드 추출 (간단한 방법)
        all_text = ' '.join(area_speeches.astype(str))
        words = re.findall(r'[가-힣]{2,}', all_text)
        
        # 빈도 계산
        word_counts = Counter(words)
        
        # 상위 키워드 저장
        top_keywords = [word for word, count in word_counts.most_common(20)]
        area_keywords[area] = top_keywords
    
    print(f"📊 정책 영역별 상위 키워드:")
    for area, keywords in area_keywords.items():
        print(f"  - {area}: {keywords[:10]}")
    
    return area_keywords

def create_comprehensive_visualizations(trend_data, area_totals, area_keywords):
    """종합 시각화 생성"""
    print("\n📊 종합 시각화 생성 중...")
    
    # 결과 디렉토리 생성
    os.makedirs('../outputs/analysis_results', exist_ok=True)
    
    # 1. 정책 영역별 발언 수 히트맵
    plt.figure(figsize=(15, 10))
    
    # 상위 8개 정책 영역만 선택
    top_areas = area_totals.head(8).index
    heatmap_data = trend_data[top_areas]
    
    sns.heatmap(heatmap_data.T, 
                annot=True, 
                fmt='.0f', 
                cmap='YlOrRd',
                cbar_kws={'label': '발언 수'})
    
    plt.title('회기별 정책 영역 발언 수 히트맵', fontsize=16, fontweight='bold')
    plt.xlabel('회기', fontsize=12)
    plt.ylabel('정책 영역', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('../outputs/analysis_results/01_policy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 정책 영역별 변화 추이
    plt.figure(figsize=(15, 8))
    
    # 상위 6개 정책 영역의 변화 추이
    top_6_areas = area_totals.head(6).index
    
    for area in top_6_areas:
        plt.plot(trend_data['session_num'], trend_data[area], 
                marker='o', linewidth=2, label=area)
    
    plt.title('회기별 정책 영역 변화 추이', fontsize=16, fontweight='bold')
    plt.xlabel('회기', fontsize=12)
    plt.ylabel('발언 수', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../outputs/analysis_results/01_policy_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 정책 영역별 총 발언 수
    plt.figure(figsize=(12, 8))
    
    plt.barh(range(len(area_totals)), area_totals.values)
    plt.yticks(range(len(area_totals)), area_totals.index)
    plt.title('정책 영역별 총 발언 수', fontsize=16, fontweight='bold')
    plt.xlabel('총 발언 수', fontsize=12)
    plt.ylabel('정책 영역', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../outputs/analysis_results/01_policy_totals.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    print("📊 종합 시각화 완료: ../outputs/analysis_results/01_policy_*.png")

def generate_insights(trend_data, area_totals, area_keywords):
    """인사이트 생성"""
    print("\n💡 정책 토픽 변화 인사이트 생성 중...")
    
    # 1. 가장 관심이 높은 정책 영역
    top_area = area_totals.index[0]
    top_count = area_totals.iloc[0]
    
    print(f"🏆 가장 관심이 높은 정책 영역: {top_area} ({top_count:,}개 발언)")
    
    # 2. 회기별 변화 패턴
    print(f"\n📈 회기별 변화 패턴:")
    
    # 각 정책 영역별 최고점 회기 찾기
    for area in area_totals.head(5).index:
        max_session = trend_data[area].idxmax()
        max_count = trend_data[area].max()
        print(f"  - {area}: {max_session}에서 최고점 ({max_count}개 발언)")
    
    # 3. 정책 우선순위 변화
    print(f"\n🎯 정책 우선순위 변화:")
    
    # 초기 회기 vs 최근 회기 비교
    available_sessions = trend_data.index.tolist()
    early_sessions = [s for s in available_sessions if '제415회' in s or '제416회' in s]
    recent_sessions = [s for s in available_sessions if '제427회' in s or '제428회' in s or '제429회' in s]
    
    if early_sessions:
        early_totals = trend_data.loc[early_sessions].sum()
    else:
        early_totals = pd.Series()
    
    if recent_sessions:
        recent_totals = trend_data.loc[recent_sessions].sum()
    else:
        recent_totals = pd.Series()
    
    if len(early_totals) > 0:
        print(f"  초기 회기 상위 3개:")
        for area in early_totals.nlargest(3).index:
            print(f"    - {area}: {early_totals[area]:,}개")
    else:
        print(f"  초기 회기 데이터 없음")
    
    if len(recent_totals) > 0:
        print(f"  최근 회기 상위 3개:")
        for area in recent_totals.nlargest(3).index:
            print(f"    - {area}: {recent_totals[area]:,}개")
    else:
        print(f"  최근 회기 데이터 없음")
    
    # 4. 정책 영역별 핵심 키워드
    print(f"\n🔑 정책 영역별 핵심 키워드:")
    for area in area_totals.head(5).index:
        if area in area_keywords:
            keywords = area_keywords[area][:5]
            print(f"  - {area}: {', '.join(keywords)}")

def main():
    """메인 실행 함수"""
    print("국회 정책 토픽 변화 추이 분석")
    print("="*80)
    
    # 1. 데이터 로드
    df = load_policy_data()
    
    # 2. 정책 발언 필터링
    policy_df = filter_policy_speeches(df)
    
    # 3. 정책 토픽 추출
    policy_df, policy_areas = extract_policy_topics(policy_df)
    
    # 4. 시계열 변화 추이 분석
    trend_data, area_totals = analyze_temporal_trends(policy_df)
    
    # 5. 정책 키워드 분석
    area_keywords = analyze_policy_keywords(policy_df)
    
    # 6. 종합 시각화 생성
    create_comprehensive_visualizations(trend_data, area_totals, area_keywords)
    
    # 7. 인사이트 생성
    generate_insights(trend_data, area_totals, area_keywords)
    
    print("\n" + "="*80)
    print("✅ 정책 토픽 변화 추이 분석 완료!")
    print("="*80)
    print("핵심 결과:")
    print("- 국회의 정책 관심사 파악")
    print("- 정책 영역별 변화 추이 분석")
    print("- 회기별 정책 우선순위 변화")
    print("- 정책 영역별 핵심 키워드")
    
    return {
        'trend_data': trend_data,
        'area_totals': area_totals,
        'area_keywords': area_keywords,
        'policy_df': policy_df
    }

if __name__ == "__main__":
    results = main()
