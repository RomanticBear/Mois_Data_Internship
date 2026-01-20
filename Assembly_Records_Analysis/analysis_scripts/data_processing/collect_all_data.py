#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
국회 회의록 전체 데이터 수집 메인 스크립트 (정당 정보 포함)
- API로 회의록 목록 수집
- HTML 파싱으로 상세 정보 추출
- 정당 정보 자동 매핑 및 추가
- CSV 파일로 저장 (party 컬럼 포함)
"""

import sys
from pathlib import Path
import os
import pandas as pd
import re
import time
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

# .env 파일 로드 (프로젝트 루트에서 찾기)
project_root = Path(__file__).resolve().parents[2]  # analysis_scripts/data_collection -> Assembly_Records_Analysis
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

from api_client import AssemblyAPIClient
from html_parser import parse_header, parse_speeches, safe_name


# 정당 정보 매핑
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


def collect_all_data(session_name: str = None, limit: int = None, test_mode: bool = False):
    """
    데이터 수집 프로세스 실행
    
    Args:
        session_name: 회차명 (None이면 모든 회차 처리)
        limit: 처리할 회의록 개수 (None이면 전체)
        test_mode: 테스트 모드 (True면 test_collected 폴더에 저장)
    """
    print(f"\n{'='*70}")
    print("📊 데이터 수집 시작")
    print(f"{'='*70}")
    if session_name:
        print(f"  대상: {session_name}")
    else:
        print(f"  대상: 모든 회차 (자동 처리)")
    print(f"  위원회: 행정안전위원회")
    
    # API 키 확인
    api_key = os.getenv("ASSEMBLY_API_KEY")
    if not api_key:
        print("  ❌ ASSEMBLY_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("  .env 파일에 ASSEMBLY_API_KEY를 추가해주세요.")
        return
    
    api_client = AssemblyAPIClient(api_key=api_key, page_size=100)
    
    dae_num = "22"
    years = ["2024", "2025"]
    
    if session_name:
        match = re.search(r"제\s*(\d+)\s*회", session_name)
        if not match:
            print(f"  ❌ 잘못된 회차명: {session_name}")
            return
    
    # API 호출
    print(f"\n[단계 2] API 호출 중... (대수: {dae_num}, 연도: {', '.join(years)})")
    
    all_records = []
    for year in years:
        year_records = api_client.search_meetings(
            dae_num=dae_num,
            conf_date=year,
            comm_name="행정안전위원회",
            max_pages=50
        )
        all_records.extend(year_records)
        print(f"  ✓ {year}년: {len(year_records)}개 수집")
    
    records = all_records
    print(f"  ✅ 총 {len(records)}개 회의록 수집 완료")
    
    # 회차 분류
    print(f"\n[단계 3] 회차 분류 중...")
    
    if session_name:
        session_records = []
        for record in records:
            if session_name in record.title:
                session_records.append(record)
        
        if not session_records:
            print(f"  ⚠️  {session_name}에 해당하는 회의록이 없습니다.")
            return
        
        sessions_dict = {session_name: session_records}
        print(f"  ✓ {session_name}: {len(session_records)}개 회의록")
    else:
        # 모든 레코드에서 회차 추출
        sessions_dict = {}
        for record in records:
            match = re.search(r"제\s*(\d+)\s*회", record.title)
            if match:
                session_key = f"제{match.group(1)}회"
                if session_key not in sessions_dict:
                    sessions_dict[session_key] = []
                sessions_dict[session_key].append(record)
        
        print(f"  ✓ 발견된 회차: {len(sessions_dict)}개")
        for session_key in sorted(sessions_dict.keys(), key=lambda x: int(re.search(r"제\s*(\d+)\s*회", x).group(1))):
            print(f"    - {session_key}: {len(sessions_dict[session_key])}개")
        
        if not sessions_dict:
            print(f"  ⚠️  회차를 찾을 수 없습니다.")
            return
    
    # 데이터 처리
    if test_mode:
        base_outdir = "../../data/test_collected"
        print(f"  🧪 테스트 모드: {base_outdir}에 저장됩니다")
    else:
        base_outdir = "../../data"
    total_created = 0
    total_failed = 0
    
    print(f"\n[단계 4] 데이터 처리 시작...")
    
    for session_key, session_records in sessions_dict.items():
        # 중복 제거
        meetings_by_id = {}
        for record in session_records:
            if record.confer_num and record.confer_num not in meetings_by_id:
                meetings_by_id[record.confer_num] = record
        
        # 제한 적용
        limited_meetings = dict(list(meetings_by_id.items())[:limit]) if limit else meetings_by_id
        
        # 회차별 폴더 생성
        session_dir = os.path.join(base_outdir, session_key)
        os.makedirs(session_dir, exist_ok=True)
        
        print(f"\n  📁 {session_key}: {len(limited_meetings)}개 회의록 처리 중...")
        
        session_created = 0
        session_failed = 0
        
        for idx, (confer_num, main_record) in enumerate(limited_meetings.items(), 1):
            try:
                # HTML URL 생성 및 파싱
                html_url = f"https://record.assembly.go.kr/assembly/viewer/minutes/xml.do?id={confer_num}&type=view"
                
                header_rows, (parsed_session, parsed_meeting_no) = parse_header(html_url)
                
                if not parsed_session:
                    if session_name:
                        parsed_session = int(re.search(r"제\s*(\d+)\s*회", session_name).group(1))
                    else:
                        parsed_session = int(re.search(r"제\s*(\d+)\s*회", session_key).group(1))
                if not parsed_meeting_no:
                    parsed_meeting_no = f"제{confer_num}호"
                
                speech_rows, _ = parse_speeches(html_url)
                
                # 정당 정보 추가
                if speech_rows:
                    speech_rows = add_party_to_speeches(speech_rows)
                
                # CSV 저장
                fname_prefix = safe_name(parsed_meeting_no)
                header_path = os.path.join(session_dir, f"{fname_prefix}_minutes_header_summary.csv")
                speech_path = os.path.join(session_dir, f"{fname_prefix}_minutes_speeches.csv")
                
                pd.DataFrame(header_rows).to_csv(header_path, index=False, encoding="utf-8-sig")
                if speech_rows:
                    pd.DataFrame(speech_rows).to_csv(speech_path, index=False, encoding="utf-8-sig")
                
                session_created += 1
                
                # 진행 상황 출력 (10개마다 또는 마지막)
                if idx % 10 == 0 or idx == len(limited_meetings):
                    print(f"    [{idx}/{len(limited_meetings)}] 처리 완료... (성공: {session_created}, 실패: {session_failed})")
                
                time.sleep(0.6)
                
            except Exception as e:
                session_failed += 1
                if idx % 10 == 0 or idx == len(limited_meetings):
                    print(f"    [{idx}/{len(limited_meetings)}] 처리 중... (실패: {session_failed})")
                continue
        
        total_created += session_created
        total_failed += session_failed
        
        print(f"  ✓ {session_key} 완료: 성공 {session_created}개, 실패 {session_failed}개")
    
    # 완료 요약
    print(f"\n{'='*70}")
    print("✅ 수집 완료")
    print(f"{'='*70}")
    print(f"  총 처리 회차: {len(sessions_dict)}개")
    print(f"  총 성공: {total_created}개")
    print(f"  총 실패: {total_failed}개")
    print(f"  저장 위치: {os.path.abspath(base_outdir)}")
    
    if os.path.exists(base_outdir):
        session_dirs = [d for d in os.listdir(base_outdir) if os.path.isdir(os.path.join(base_outdir, d))]
        print(f"\n  생성된 회차 폴더: {len(session_dirs)}개")
        for session_dir_name in sorted(session_dirs):
            session_path = os.path.join(base_outdir, session_dir_name)
            file_count = len([f for f in os.listdir(session_path) if f.endswith('.csv')]) // 2
            print(f"    - {session_dir_name}: {file_count}개 회의록")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="국회 회의록 전체 데이터 수집")
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="수집할 회차 (예: 제418회). 지정하지 않으면 모든 회차 자동 처리"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 회의록 개수 (지정하지 않으면 전체)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="테스트 모드 (test_collected 폴더에 저장)"
    )
    args = parser.parse_args()
    collect_all_data(args.session, args.limit, test_mode=args.test)

