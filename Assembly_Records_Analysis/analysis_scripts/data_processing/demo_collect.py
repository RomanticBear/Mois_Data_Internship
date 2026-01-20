#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 적재 과정 시연용 데모 스크립트
- 소수의 데이터만 빠르게 처리
- 핵심 과정을 명확하게 출력
- 결과는 demo_output 폴더에 저장
"""

import sys
from pathlib import Path
import os
import pandas as pd
import re
import time
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

# .env 파일 로드
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

from api_client import AssemblyAPIClient
from html_parser import parse_header, parse_speeches, safe_name

# 정당 정보 매핑 (일부만 사용)
PARTY_MAPPING = {
    '신정훈': '더불어민주당',
    '윤건영': '더불어민주당',
    '서범수': '국민의힘',
    '권칠승': '더불어민주당',
    '김성회': '더불어민주당',
    '모경종': '더불어민주당',
    '박정현': '더불어민주당',
    '양부남': '더불어민주당',
    '위성곤': '더불어민주당',
    '이광희': '더불어민주당',
    '이상식': '더불어민주당',
    '이해식': '더불어민주당',
    '채현일': '더불어민주당',
    '한병도': '더불어민주당',
    '고동진': '국민의힘',
    '박덕흠': '국민의힘',
    '박수민': '국민의힘',
    '이달희': '국민의힘',
    '이성권': '국민의힘',
    '주호영': '국민의힘',
    '정춘생': '조국혁신당',
    '용혜인': '기본소득당'
}


def add_party_to_speeches(speech_rows):
    """발언 데이터에 정당 정보 추가"""
    if not speech_rows:
        return speech_rows
    
    df = pd.DataFrame(speech_rows)
    df['party'] = df['speaker_name'].map(PARTY_MAPPING).fillna('미분류')
    return df.to_dict('records')


def print_step(step_num, title, detail=""):
    """단계 출력 헬퍼 함수"""
    print(f"\n[단계 {step_num}] {title}")
    if detail:
        print(f"  {detail}")


def demo_collect(demo_count: int = 2, session_name: str = "제420회"):
    """
    시연용 데이터 수집 데모
    
    Args:
        demo_count: 처리할 회의록 개수 (기본: 2개)
        session_name: 대상 회차 (기본: 제420회)
    """
    print("\n" + "="*70)
    print("데이터 적재 과정 시연 (데모 모드)")
    print("="*70)
    print(f"대상 회차: {session_name}")
    print(f"처리할 회의록 수: {demo_count}개")
    print(f"저장 폴더: demo_output/")
    print("="*70)
    
    # 단계 1: 초기화
    print_step(1, "초기화", "API 클라이언트 생성 및 설정 확인")
    
    api_key = os.getenv("ASSEMBLY_API_KEY")
    if not api_key:
        print("  [오류] ASSEMBLY_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("  .env 파일에 ASSEMBLY_API_KEY를 추가해주세요.")
        return
    
    api_client = AssemblyAPIClient(api_key=api_key, page_size=100)
    print("  API 클라이언트 생성 완료")
    
    dae_num = "22"
    years = ["2024", "2025"]
    print(f"  설정: 대수={dae_num}, 연도={', '.join(years)}")
    
    # 단계 2: API 호출
    print_step(2, "API 호출", "행정안전위원회 회의록 목록 조회 중...")
    
    all_records = []
    for year in years:
        print(f"  {year}년 데이터 조회 중...", end=" ")
        year_records = api_client.search_meetings(
            dae_num=dae_num,
            conf_date=year,
            comm_name="행정안전위원회",
            max_pages=2  # 데모용으로 페이지 수 제한
        )
        all_records.extend(year_records)
        print(f"완료 ({len(year_records)}개)")
    
    print(f"\n  총 {len(all_records)}개 회의록 목록 수집 완료")
    
    if not all_records:
        print("  [경고] 수집된 회의록이 없습니다.")
        return
    
    # 단계 3: 회차별 필터링 및 데이터 선택
    print_step(3, "회차별 필터링", f"{session_name} 회의록 선택 중...")
    
    # 해당 회차의 레코드만 필터링
    session_records = []
    for record in all_records:
        if session_name in record.title:
            session_records.append(record)
    
    if not session_records:
        print(f"  [경고] {session_name}에 해당하는 회의록이 없습니다.")
        return
    
    print(f"  {session_name} 회의록: {len(session_records)}개 발견")
    
    # 중복 제거
    meetings_by_id = {}
    for record in session_records:
        if record.confer_num and record.confer_num not in meetings_by_id:
            meetings_by_id[record.confer_num] = record
    
    # 데모용으로 제한된 수만큼만 선택
    selected_meetings = dict(list(meetings_by_id.items())[:demo_count])
    
    print(f"  처리할 회의록 {len(selected_meetings)}개 선택 완료")
    print(f"\n  처리할 회의록 목록:")
    for idx, (confer_num, record) in enumerate(selected_meetings.items(), 1):
        print(f"    {idx}. [{confer_num}] {record.title[:50]}")
        print(f"        날짜: {record.conf_date}")
    
    # 출력 디렉토리 설정
    base_outdir = "../../data/demo_output"
    session_dir = os.path.join(base_outdir, session_name)
    os.makedirs(session_dir, exist_ok=True)
    
    print(f"\n  저장 경로: {os.path.abspath(session_dir)}")
    
    # 단계 4: HTML 파싱 및 데이터 추출
    print_step(4, "HTML 파싱", "회의록 상세 정보 추출 중...")
    
    total_created = 0
    total_failed = 0
    
    for idx, (confer_num, main_record) in enumerate(selected_meetings.items(), 1):
        print(f"\n  [{idx}/{len(selected_meetings)}] 회의록 처리 중...")
        print(f"  회의번호: {confer_num} | {main_record.title[:40]}")
        
        try:
            # HTML URL 생성
            html_url = f"https://record.assembly.go.kr/assembly/viewer/minutes/xml.do?id={confer_num}&type=view"
            
            # 헤더 파싱
            print(f"  - 헤더 정보 추출 중...", end=" ")
            header_rows, (parsed_session, parsed_meeting_no) = parse_header(html_url)
            print(f"완료 ({len(header_rows)}행)")
            
            if not parsed_session and session_name:
                parsed_session = int(re.search(r"제\s*(\d+)\s*회", session_name).group(1))
            if not parsed_meeting_no:
                parsed_meeting_no = f"제{confer_num}호"
            
            # 발언 파싱
            print(f"  - 발언 정보 추출 중...", end=" ")
            speech_rows, _ = parse_speeches(html_url)
            print(f"완료 ({len(speech_rows)}개 발언)")
            
            # 정당 정보 추가
            if speech_rows:
                print(f"  - 정당 정보 매핑 중...", end=" ")
                speech_rows = add_party_to_speeches(speech_rows)
                # 정당 분포 출력
                df_temp = pd.DataFrame(speech_rows)
                party_counts = df_temp['party'].value_counts()
                party_str = ", ".join([f"{p}: {c}" for p, c in party_counts.head(3).items()])
                print(f"완료 ({party_str})")
            
            # CSV 저장
            fname_prefix = safe_name(parsed_meeting_no)
            header_path = os.path.join(session_dir, f"{fname_prefix}_minutes_header_summary.csv")
            speech_path = os.path.join(session_dir, f"{fname_prefix}_minutes_speeches.csv")
            
            pd.DataFrame(header_rows).to_csv(header_path, index=False, encoding="utf-8-sig")
            if speech_rows:
                pd.DataFrame(speech_rows).to_csv(speech_path, index=False, encoding="utf-8-sig")
            
            print(f"  - CSV 저장 완료: {fname_prefix}_*.csv")
            
            total_created += 1
            time.sleep(0.3)  # 데모용으로 지연 시간 단축
            
        except Exception as e:
            print(f"  [오류] 처리 실패: {str(e)[:80]}")
            total_failed += 1
            continue
    
    # 완료 요약
    print_step(5, "완료", "데이터 수집 요약")
    
    print(f"\n처리 완료")
    print(f"  성공: {total_created}개")
    print(f"  실패: {total_failed}개")
    print(f"  저장 위치: {os.path.abspath(session_dir)}")
    
    if total_created > 0:
        # 파일 목록 출력
        print(f"\n생성된 파일:")
        if os.path.exists(session_dir):
            files = [f for f in os.listdir(session_dir) if f.endswith('.csv')]
            for f in sorted(files):
                file_path = os.path.join(session_dir, f)
                file_size = os.path.getsize(file_path)
                print(f"  - {f} ({file_size:,} bytes)")
    
    print(f"\n{'='*70}")
    print("데모 완료")
    print("="*70)
    print(f"\n전체 데이터 수집: python collect_all_data.py --session {session_name}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="데이터 적재 과정 시연용 데모 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python demo_collect.py              # 기본 2개 처리
  python demo_collect.py --count 3    # 3개 처리
        """
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="처리할 회의록 개수 (기본: 2개)"
    )
    parser.add_argument(
        "--session",
        type=str,
        default="제420회",
        help="대상 회차 (기본: 제420회)"
    )
    
    args = parser.parse_args()
    
    demo_collect(demo_count=args.count, session_name=args.session)

