#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""분석 결과 요약 출력"""

import json
from pathlib import Path

def main():
    result_file = Path("analysis_results/제415회_openai_analysis.json")
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("📊 OpenAI API 분석 결과 출력")
    print("=" * 80)
    
    # 기본 정보
    print(f"\n📋 기본 정보:")
    print(f"  - 회차: {data['session_name']}")
    print(f"  - 총 발언: {data['total_speeches']:,}개")
    print(f"  - 품질 발언: {data['quality_speeches']:,}개 ({data['quality_speeches']/data['total_speeches']*100:.1f}%)")
    print(f"  - 분석 버전: {data['analysis_version']}")
    print(f"  - 분석 시간: {data['analysis_timestamp']}")
    
    # 핵심 이슈
    print(f"\n🔍 핵심 이슈 ({len(data['session_summary']['key_issues'])}개):")
    for i, issue in enumerate(data['session_summary']['key_issues'], 1):
        print(f"\n  {i}. [{issue['importance']}] {issue['issue']}")
        print(f"     설명: {issue['description']}")
        print(f"     언급 정당: {', '.join(issue['mentioned_parties'])}")
    
    # 세션 요약
    print(f"\n📝 세션 요약:")
    print(f"  {data['session_summary']['raw_summary']}")
    
    # 정당별 입장
    print(f"\n👥 정당별 입장 개요:")
    for party, info in data['session_summary']['metadata']['party_positions_overview'].items():
        print(f"\n  - {party}: {info['stance']}")
        print(f"    주요 관심사: {', '.join(info['main_concerns'])}")
        print(f"    주요 발언: {info['key_statements']}")
    
    # 주요 쟁점
    print(f"\n⚔️ 주요 쟁점:")
    for conflict in data['session_summary']['metadata']['major_conflicts']:
        print(f"  - {conflict['topic']}: {conflict['nature']}")
        print(f"    참여 정당: {', '.join(conflict['parties_involved'])}")
    
    # 주요 사건
    print(f"\n📰 주요 사건:")
    for event in data['session_summary']['metadata']['key_events']:
        print(f"  - {event['event']}")
        print(f"    설명: {event['description']}")
        print(f"    국회 대응: {event['response']}")
    
    # 안건별 분석
    print(f"\n📋 안건별 정당 입장 분석 ({len(data['party_positions'])}개 안건):")
    for i, agenda in enumerate(data['party_positions'], 1):
        print(f"\n  {i}. 안건: {agenda['agenda_title']}")
        print(f"     협력 수준: {agenda['cooperation_level']}")
        print(f"     합의점: {', '.join(agenda['consensus_points'])}")
        print(f"     대립점: {', '.join(agenda['conflict_points'])}")
        print(f"     정당별 입장 ({len(agenda['party_positions'])}개 정당):")
        for pos in agenda['party_positions']:
            print(f"       - {pos['party_name']}: {pos['stance_label']}")
            print(f"         주요 포인트: {', '.join(pos['key_points'])}")
            print(f"         우려사항: {', '.join(pos['concerns'])}")
            print(f"         제안사항: {', '.join(pos['suggestions'])}")
            print(f"         요약: {pos['summary_text']}")
    
    # QA 효과성 분석
    print(f"\n💬 QA 효과성 분석:")
    qa = data['qa_analysis']
    print(f"  - 질의-응답 쌍: {qa['total_qa_pairs']}개")
    print(f"  - 품질 분포:")
    for level, value in qa['quality_distribution'].items():
        print(f"    {level}: {value}")
    print(f"  - 질문 유형:")
    for qtype, value in qa['question_types'].items():
        print(f"    {qtype}: {value}")
    print(f"  - 답변 품질:")
    for metric, value in qa['answer_quality'].items():
        print(f"    {metric}: {value}")
    print(f"  - 주요 이슈:")
    for issue in qa['key_issues']:
        print(f"    - {issue['issue']}: {issue['qa_count']}개 질의-응답, 품질 {issue['quality']}")
    print(f"  - 개선 제안:")
    for suggestion in qa['improvement_suggestions']:
        print(f"    - {suggestion}")
    
    print("\n" + "=" * 80)
    print("✅ 분석 결과 출력 완료")
    print("=" * 80)

if __name__ == "__main__":
    main()


