#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI 분석 결과 확인 스크립트
"""

import json
import os
from pathlib import Path

def view_analysis_results(session_name: str = "제415회"):
    """분석 결과를 보기 좋게 출력"""
    
    # 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    json_path = os.path.join(project_root, 'analysis_results', f'{session_name}_openai_analysis.json')
    
    if not os.path.exists(json_path):
        print(f"❌ 결과 파일을 찾을 수 없습니다: {json_path}")
        return
    
    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 70)
    print(f"📊 {session_name} OpenAI 분석 결과")
    print("=" * 70)
    
    # 기본 정보
    print(f"\n📈 기본 통계")
    print(f"  - 총 발언 수: {data['total_speeches']:,}개")
    print(f"  - 품질 발언 수: {data['quality_speeches']:,}개 ({data['quality_speeches']/data['total_speeches']*100:.1f}%)")
    print(f"  - 분석 시간: {data['analysis_timestamp']}")
    
    # 회차 요약
    if 'session_summary' in data and data['session_summary']:
        summary = data['session_summary']
        print(f"\n🔍 핵심 이슈 ({len(summary.get('key_issues', []))}개)")
        for i, issue in enumerate(summary.get('key_issues', []), 1):
            print(f"  {i}. {issue['issue']} ({issue['importance']})")
            print(f"     설명: {issue['description']}")
            print(f"     언급 정당: {', '.join(issue.get('mentioned_parties', []))}")
        
        # 정당별 입장
        print(f"\n👥 정당별 주요 관심사")
        for party, info in summary.get('party_positions', {}).items():
            print(f"  - {party}:")
            if 'main_concerns' in info:
                print(f"    관심사: {', '.join(info['main_concerns'])}")
            if 'key_statements' in info:
                print(f"    주요 발언: {info['key_statements']}")
        
        # 주요 쟁점
        if 'major_conflicts' in summary:
            print(f"\n⚔️ 주요 쟁점 ({len(summary['major_conflicts'])}개)")
            for conflict in summary['major_conflicts']:
                print(f"  - {conflict['topic']}")
                print(f"    관련 정당: {', '.join(conflict.get('parties_involved', []))}")
                print(f"    성격: {conflict.get('nature', 'N/A')}")
        
        # 주요 사건
        if 'key_events' in summary:
            print(f"\n📰 주요 사건")
            for event in summary['key_events']:
                print(f"  - {event['event']}")
                print(f"    설명: {event['description']}")
                print(f"    국회 대응: {event['response']}")
        
        # 회차 특징
        if 'session_characteristics' in summary:
            print(f"\n💡 회차 특징")
            print(f"  {summary['session_characteristics']}")
    
    # 안건별 정당 입장
    if 'party_positions' in data and data['party_positions']:
        print(f"\n📋 안건별 정당 입장 비교 ({len(data['party_positions'])}개 안건)")
        for agenda, info in data['party_positions'].items():
            agenda_short = agenda[:50] + "..." if len(agenda) > 50 else agenda
            print(f"\n  안건: {agenda_short}")
            print(f"  협력 수준: {info.get('cooperation_level', 'N/A')}")
            
            if 'consensus_points' in info:
                print(f"  ✅ 합의점:")
                for point in info['consensus_points']:
                    print(f"    - {point}")
            
            if 'conflict_points' in info:
                print(f"  ⚠️ 대립점:")
                for point in info['conflict_points']:
                    print(f"    - {point}")
            
            if 'party_positions' in info:
                print(f"  정당별 입장:")
                for party, pos in info['party_positions'].items():
                    print(f"    - {party}: {pos.get('position', 'N/A')}")
                    if 'key_points' in pos:
                        print(f"      주요 포인트: {', '.join(pos['key_points'][:2])}")
    
    # 질의-응답 분석
    if 'qa_analysis' in data and data['qa_analysis']:
        qa = data['qa_analysis']
        print(f"\n💬 질의-응답 효과성 분석")
        print(f"  - 총 질의-응답 쌍: {qa.get('qa_pairs_count', 0)}개")
        
        if 'quality_distribution' in qa:
            quality = qa['quality_distribution']
            print(f"  - 응답 품질 분포:")
            print(f"    고품질: {quality.get('high', 0)}%")
            print(f"    중품질: {quality.get('medium', 0)}%")
            print(f"    저품질: {quality.get('low', 0)}%")
        
        if 'answer_quality' in qa:
            ans_quality = qa['answer_quality']
            print(f"  - 응답 품질 점수:")
            print(f"    완성도: {ans_quality.get('completeness', 0)}/10")
            print(f"    구체성: {ans_quality.get('specificity', 0)}/10")
            print(f"    응답성: {ans_quality.get('responsiveness', 0)}/10")
        
        if 'question_types' in qa:
            q_types = qa['question_types']
            print(f"  - 질문 유형:")
            print(f"    정책 질의: {q_types.get('policy_inquiry', 0)}%")
            print(f"    사실 확인: {q_types.get('fact_checking', 0)}%")
            print(f"    비판 질의: {q_types.get('criticism', 0)}%")
            print(f"    제안 질의: {q_types.get('suggestion', 0)}%")
        
        if 'improvement_suggestions' in qa:
            print(f"  - 개선 제안:")
            for suggestion in qa['improvement_suggestions']:
                print(f"    • {suggestion}")
    
    # 생성된 파일 목록
    print(f"\n📁 생성된 파일")
    results_dir = os.path.join(project_root, 'analysis_results')
    files = [
        f'{session_name}_openai_analysis.json',
        f'{session_name}_key_issues.png',
        f'{session_name}_party_concerns.png',
        f'{session_name}_qa_quality.png'
    ]
    
    for file in files:
        file_path = os.path.join(results_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            size_kb = size / 1024
            print(f"  ✅ {file} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ {file} (파일 없음)")
    
    print("\n" + "=" * 70)
    print(f"💡 팁: 시각화 파일은 analysis_results/ 폴더에서 확인하세요!")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    session_name = sys.argv[1] if len(sys.argv) > 1 else "제415회"
    view_analysis_results(session_name)






