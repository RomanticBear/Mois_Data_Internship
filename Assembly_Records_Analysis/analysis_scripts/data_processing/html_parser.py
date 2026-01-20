# -*- coding: utf-8 -*-
"""
HTML 파싱 유틸리티 모듈
- 회의록 HTML에서 헤더/발언 정보 추출
- parse_header, parse_speeches 등 파싱 함수 제공
- collect_all_data.py에서 사용
"""

import sys
from pathlib import Path
import os
import pandas as pd
import re
import time
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

# .env 파일 로드 (프로젝트 루트에서 찾기)
project_root = Path(__file__).resolve().parents[1]  # analysis_scripts/data_collection -> Assembly_Records_Analysis
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

from api_client import AssemblyAPIClient

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MOIS-Intern-Project/1.0)"}
SLEEP_SEC = 0.6


def clean(s: str) -> str:
    """텍스트 정리"""
    if not s:
        return ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def get_soup(url: str) -> BeautifulSoup:
    """HTML/XML 뷰어 URL에서 Soup 객체 생성"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=25)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        if not soup.select_one("#minutes"):
            print(f"  ⚠️  '#minutes' 섹션을 찾지 못했습니다.")
        return soup
    except Exception as e:
        print(f"  ❌ request error: {e}")
        raise


def parse_header_title(soup: BeautifulSoup):
    """회의록 헤더 정보 파싱 (기존 크롤러 함수 재사용)"""
    session, session_type, meeting_no = None, "", ""
    tit_wrap = soup.select_one("#minutes .minutes_header .tit_wrap")
    if tit_wrap:
        turn = tit_wrap.select_one("p.turn")
        if turn:
            t = turn.get_text("\n", strip=True)
            m1 = re.search(r"제\s*(\d+)\s*회", t)
            if m1: session = int(m1.group(1))
            m2 = re.search(r"\(([^)]+)\)", t)
            if m2: session_type = clean(m2.group(1))
        pnum = tit_wrap.select_one("p.num")
        if pnum:
            meeting_no = clean(pnum.get_text())
        if not meeting_no:
            header_text = clean(soup.select_one("#minutes .minutes_header").get_text(" ")) \
                          if soup.select_one("#minutes .minutes_header") else ""
            m3 = re.search(r"제\s*\d+\s*[차호]", header_text)
            if m3: meeting_no = m3.group(0)
    return session, session_type, meeting_no


def parse_date_only(soup: BeautifulSoup):
    """날짜 파싱 (기존 크롤러 함수 재사용)"""
    date_text = ""
    for li in soup.select("#minutes .minutes_header .place ul > li"):
        sbj_el = li.select_one(".sbj, .sbj.lts2, .sbj.lts4")
        sbj = clean(sbj_el.get_text()) if sbj_el else ""
        if "일시" in sbj:
            con_el = li.select_one("p.con")
            if con_el:
                date_text = clean(con_el.get_text())
            break
    return date_text


def get_text_with_br(container: Tag) -> str:
    """BR 태그를 줄바꿈으로 변환 (기존 크롤러 함수 재사용)"""
    if container is None: return ""
    parts = []
    for node in container.descendants:
        if isinstance(node, NavigableString):
            parts.append(str(node))
        elif isinstance(node, Tag) and node.name.lower() == "br":
            parts.append("\n")
    text = "".join(parts)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _parse_agenda_title(a_or_p: Tag):
    """안건 제목 파싱 (기존 크롤러 함수 재사용)"""
    txt = ""
    if a_or_p:
        txt = clean(a_or_p.get_text())
    m = re.match(r"(\d+)\.\s*(.+)", txt)
    if m:
        return m.group(1), clean(m.group(2))
    return "", txt


def _nearest_time_after(node: Tag):
    """가장 가까운 시간 찾기 (기존 크롤러 함수 재사용)"""
    cur = node
    for _ in range(12):
        cur = cur.find_next()
        if not isinstance(cur, Tag):
            continue
        cls = cur.get("class", [])
        if cur.name == "p" and "tit_sm" in cls and "taR" in cls:
            return clean(cur.get_text())
        if cur.name == "p" and "tit_sm" in cls and "angun" in cls:
            break
        if cur.name == "div" and "speaker" in cls:
            break
    return ""


def _extract_agenda_groups(soup: BeautifulSoup):
    """안건 그룹 추출 (기존 크롤러 함수 재사용)"""
    speakers = soup.select("#minutes .minutes_body .speaker")
    bars = soup.select("#minutes .minutes_body p.tit_sm.angun")
    
    groups = []
    i = 0
    while i < len(bars):
        cur_bar = bars[i]
        orders, titles, times = [], [], []
        
        while True:
            a_tit = cur_bar.select_one("a.tit")
            num, title = _parse_agenda_title(a_tit if a_tit else cur_bar)
            if num: orders.append(num)
            if title: titles.append(title)
            tm = _nearest_time_after(cur_bar)
            if tm: times.append(tm)
            
            next_bar = bars[i+1] if i+1 < len(bars) else None
            if not next_bar:
                break
            found_speaker_between = False
            probe = cur_bar
            while True:
                probe = probe.find_next()
                if not probe or probe is next_bar:
                    break
                if isinstance(probe, Tag) and "speaker" in probe.get("class", []):
                    found_speaker_between = True
                    break
            if found_speaker_between:
                break
            i += 1
            cur_bar = next_bar
        
        first_spk_after_group = cur_bar.find_next(class_="speaker")
        start_order = None
        if first_spk_after_group:
            for idx, spk in enumerate(soup.select("#minutes .minutes_body .speaker"), start=1):
                if spk is first_spk_after_group:
                    start_order = idx
                    break
        if start_order is not None:
            groups.append({
                "start_order": start_order,
                "orders": ",".join(orders) if orders else "",
                "titles": " / ".join(titles) if titles else "",
                "times": " / ".join(times) if times else "",
            })
        
        i += 1
    
    groups.sort(key=lambda g: g["start_order"])
    return groups, soup.select("#minutes .minutes_body .speaker")


def parse_header(url: str):
    """헤더 파싱 (기존 크롤러 함수 재사용)"""
    soup = get_soup(url)
    session, session_type, meeting_no = parse_header_title(soup)
    date_text = parse_date_only(soup)
    
    rows = []
    rows.append({
        "session": session, "session_type": session_type, "meeting_no": meeting_no,
        "date": date_text, "section": "일시", "item_order": "", "item_text": date_text,
    })
    
    for li in soup.select("#minutes .minutes_header .place ul > li"):
        sbj_el = li.select_one(".sbj, .sbj.lts4, .sbj.lts2")
        sbj = clean(sbj_el.get_text()) if sbj_el else ""
        if "의사일정" in sbj or "상정된 안건" in sbj:
            label = "의사일정" if "의사일정" in sbj else "상정된 안건"
            items = li.select("ul.list_num li")
            for idx, it in enumerate(items, start=1):
                a = it.select_one("a")
                txt = clean(a.get_text()) if a and clean(a.get_text()) else clean(it.get_text())
                txt = re.sub(r"^\d+\.\s*", "", txt)
                rows.append({
                    "session": session, "session_type": session_type, "meeting_no": meeting_no,
                    "date": date_text, "section": label, "item_order": idx, "item_text": txt,
                })
    
    return rows, (session, meeting_no)


def parse_speeches(url: str):
    """발언 파싱 (기존 크롤러 함수 재사용)"""
    soup = get_soup(url)
    session, session_type, meeting_no = parse_header_title(soup)
    date_text = parse_date_only(soup)
    
    groups, speakers = _extract_agenda_groups(soup)
    
    ranges = []
    for idx, g in enumerate(groups):
        start = g["start_order"]
        end = (groups[idx+1]["start_order"] - 1) if idx+1 < len(groups) else len(speakers)
        ranges.append((start, end, g))
    
    rows = []
    for order, spk in enumerate(speakers, start=1):
        data_mem_id = ""
        for key in ("data-mem_id", "data_mem_id", "data-mem-id"):
            if spk.has_attr(key):
                data_mem_id = spk.get(key) or ""
                break
        
        name_el = spk.select_one(".man .txt strong.name")
        pos_el  = spk.select_one(".man .txt .position")
        area_el = spk.select_one(".man .txt .area")
        
        name     = clean(name_el.get_text()) if name_el else clean(spk.get("data-name", ""))
        position = clean(pos_el.get_text())  if pos_el  else clean(spk.get("data-pos", ""))
        area     = clean(area_el.get_text()) if area_el else ""
        
        speech = get_text_with_br(spk.select_one(".talk .txt"))
        
        ag_orders = ag_titles = ag_times = ""
        for start, end, g in ranges:
            if start <= order <= end:
                ag_orders = g["orders"]
                ag_titles = g["titles"]
                ag_times  = g["times"]
                break
        
        rows.append({
            "session": session,
            "session_type": session_type,
            "meeting_no": meeting_no,
            "date": date_text,
            "speech_order": order,
            "speaker_name": name,
            "speaker_position": position,
            "speaker_area": area,
            "data_mem_id": data_mem_id,
            "agenda_item_orders": ag_orders,
            "agenda_item_titles": ag_titles,
            "agenda_item_times": ag_times,
            "speech_text": speech,
        })
    return rows, (session, meeting_no)


def safe_name(s: str) -> str:
    """파일명 안전화"""
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    s = s.replace("?", "").replace("*", "").replace('"', "").replace("<", "").replace(">", "").replace("|", "")
    return s.strip()


def collect_session_with_html(session_name: str, api_key: str = None, limit: int = 3, verbose: bool = True):
    """
    HTML 파싱을 사용한 회차별 데이터 수집
    
    Args:
        session_name: 회차명 (예: "제418회")
        api_key: API 인증키 (None이면 환경 변수에서 읽음)
        limit: 테스트용 최대 수집 개수
        verbose: 상세 로그 출력 여부
    """
    # API 키 가져오기 (환경 변수 우선, 없으면 파라미터 사용)
    if not api_key:
        api_key = os.getenv("ASSEMBLY_API_KEY")
        if not api_key:
            print("❌ ASSEMBLY_API_KEY 환경 변수가 설정되지 않았습니다.")
            print("   .env 파일에 ASSEMBLY_API_KEY를 추가하거나 api_key 파라미터를 전달해주세요.")
            return
    print("=" * 70)
    print(f"📊 HTML 파싱을 사용한 데이터 수집: {session_name}")
    print("=" * 70)
    
    if verbose:
        print("\n[단계 1] 초기화")
        print("  - API 클라이언트 생성")
        print("  - 출력 디렉토리: ../../data")
    
    # 출력 디렉토리
    base_outdir = "../../data"
    os.makedirs(base_outdir, exist_ok=True)
    
    # API 클라이언트
    api_client = AssemblyAPIClient(api_key=api_key, page_size=100)
    
    # 회차 번호 추출
    match = re.search(r"제\s*(\d+)\s*회", session_name)
    if not match:
        print(f"❌ 잘못된 회차명 형식: {session_name}")
        return
    
    session_num = int(match.group(1))
    dae_num = "22"
    # 22대 국회는 2024년, 2025년에 해당
    years = ["2024", "2025"]
    
    if verbose:
        print(f"\n[단계 2] API 검색")
        print(f"  - 대수: {dae_num}")
        print(f"  - 연도: {', '.join(years)} (22대는 2024, 2025년)")
        print(f"  - 위원회: 행정안전위원회")
        print(f"  - 회차: {session_name}")
        print("\n  🔍 API 호출 중...")
    else:
        print(f"\n🔍 API 검색 중... (대수: {dae_num}, 연도: {', '.join(years)})")
    
    try:
        # API로 회의록 검색 (행정안전위원회만 필터링)
        # 2024, 2025년 모두 조회
        all_records = []
        for year in years:
            if verbose:
                print(f"  📅 {year}년 데이터 조회 중...")
            year_records = api_client.search_meetings(
                dae_num=dae_num,
                conf_date=year,
                comm_name="행정안전위원회",  # 행정안전위원회만 필터링
                max_pages=50
            )
            all_records.extend(year_records)
            if verbose:
                print(f"     ✅ {year}년: {len(year_records)}개 수집")
        
        records = all_records
        
        print(f"✅ 총 {len(records)}개의 회의록 수집 완료 (행정안전위원회)")
        
        if verbose and records:
            print(f"\n  📋 API 반환 결과 샘플 (처음 3개):")
            for i, record in enumerate(records[:3], 1):
                print(f"    {i}. 회의번호: {record.confer_num}")
                print(f"       제목: {record.title[:50]}...")
                print(f"       날짜: {record.conf_date}")
                print(f"       위원회: {record.comm_name}")
        
        # 해당 회차의 레코드만 필터링
        if verbose:
            print(f"\n[단계 3] 회차 필터링")
            print(f"  - 검색 조건: '{session_name}' 포함")
        
        session_records = []
        for record in records:
            if session_name in record.title:
                session_records.append(record)
        
        print(f"✅ {session_name}에 해당하는 회의록: {len(session_records)}개")
        
        if verbose and session_records:
            print(f"\n  📋 필터링된 회의록 목록:")
            for i, record in enumerate(session_records[:5], 1):
                print(f"    {i}. 회의번호: {record.confer_num} | {record.title[:60]}")
            if len(session_records) > 5:
                print(f"    ... 외 {len(session_records) - 5}개")
        
        if not session_records:
            print(f"⚠️  {session_name}에 해당하는 회의록이 없습니다.")
            return
        
        # 제한 적용
        if limit:
            session_records = session_records[:limit]
            print(f"📝 테스트 모드: {limit}개만 처리")
        
        # 세션 디렉토리 생성
        session_dir = os.path.join(base_outdir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        # confer_num별로 그룹화 (중복 제거)
        if verbose:
            print(f"\n[단계 4] 중복 제거 및 그룹화")
        
        meetings_by_id = {}
        
        for record in session_records:
            # confer_num으로 그룹화 (같은 회의번호의 회의록은 하나만 처리)
            if record.confer_num and record.confer_num not in meetings_by_id:
                meetings_by_id[record.confer_num] = record
        
        print(f"📋 고유 회의록: {len(meetings_by_id)}개")
        
        if verbose:
            print(f"\n  📋 처리할 회의번호 목록:")
            for i, (confer_num, record) in enumerate(list(meetings_by_id.items())[:10], 1):
                print(f"    {i}. 회의번호: {confer_num} | {record.title[:50]}...")
            if len(meetings_by_id) > 10:
                print(f"    ... 외 {len(meetings_by_id) - 10}개")
        
        print(f"\n📝 회의록 처리 시작 (HTML 파싱)...")
        
        created_count = 0
        failed_count = 0
        
        for idx, (confer_num, main_record) in enumerate(meetings_by_id.items(), 1):
            try:
                # HTML/XML 뷰어 URL 생성 (기존 크롤러 방식)
                # confer_num을 사용 (이전 테스트에서 이 방식이 작동함)
                meeting_id = str(confer_num)
                
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"[{idx}/{len(meetings_by_id)}] 회의록 처리 중")
                    print(f"{'='*70}")
                    print(f"  제목: {main_record.title}")
                    print(f"  회의번호: {confer_num}")
                    print(f"  날짜: {main_record.conf_date}")
                    print(f"  위원회: {main_record.comm_name}")
                else:
                    print(f"\n[{idx}/{len(meetings_by_id)}] 처리 중: {main_record.title[:60]}...")
                
                html_url = f"https://record.assembly.go.kr/assembly/viewer/minutes/xml.do?id={meeting_id}&type=view"
                
                if verbose:
                    print(f"  🌐 HTML 뷰어 URL:")
                    print(f"     {html_url}")
                
                # 헤더 파싱
                if verbose:
                    print(f"  📄 HTML 다운로드 및 파싱 중...")
                
                header_rows, (parsed_session, parsed_meeting_no) = parse_header(html_url)
                
                if not parsed_session:
                    parsed_session = session_num
                if not parsed_meeting_no:
                    parsed_meeting_no = f"제{meeting_id}호"
                
                if verbose:
                    print(f"  ✅ 헤더 파싱 완료: {len(header_rows)}행")
                    print(f"     - 회차: {parsed_session}, 회의번호: {parsed_meeting_no}")
                
                # 발언 파싱
                speech_rows, _ = parse_speeches(html_url)
                
                if verbose:
                    print(f"  ✅ 발언 파싱 완료: {len(speech_rows)}개")
                    if speech_rows:
                        print(f"     - 첫 번째 발언 샘플:")
                        first_speech = speech_rows[0]
                        print(f"       발언자: {first_speech.get('speaker_name', 'N/A')}")
                        print(f"       내용: {first_speech.get('speech_text', '')[:100]}...")
                
                # 파일명 생성
                fname_prefix = safe_name(parsed_meeting_no)
                
                # CSV 저장
                header_path = os.path.join(session_dir, f"{fname_prefix}_minutes_header_summary.csv")
                speech_path = os.path.join(session_dir, f"{fname_prefix}_minutes_speeches.csv")
                
                if verbose:
                    print(f"  💾 CSV 저장 중...")
                    print(f"     - 헤더: {header_path}")
                    print(f"     - 발언: {speech_path}")
                
                pd.DataFrame(header_rows).to_csv(header_path, index=False, encoding="utf-8-sig")
                
                if speech_rows:
                    pd.DataFrame(speech_rows).to_csv(speech_path, index=False, encoding="utf-8-sig")
                    if verbose:
                        print(f"  ✅ 저장 완료!")
                        print(f"     - 헤더: {len(header_rows)}행")
                        print(f"     - 발언: {len(speech_rows)}개")
                    else:
                        print(f"  ✅ 완료: 헤더 {len(header_rows)}행, 발언 {len(speech_rows)}개")
                else:
                    print(f"  ⚠️  발언 데이터 없음")
                
                created_count += 1
                time.sleep(SLEEP_SEC)
                
            except Exception as e:
                print(f"  ❌ 처리 실패: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
        
        print(f"\n{'='*70}")
        print(f"✅ 수집 완료!")
        print(f"   - 성공: {created_count}개")
        print(f"   - 실패: {failed_count}개")
        print(f"\n📁 저장 위치: {os.path.abspath(session_dir)}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """
    참고: 이 파일은 유틸리티 모듈입니다.
    메인 데이터 수집은 collect_all_data.py를 사용하세요:
    
        python collect_all_data.py              # 모든 회차 수집
        python collect_all_data.py --session 제418회  # 특정 회차만
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HTML 파싱을 사용한 회차별 데이터 수집 (테스트/디버깅용)",
        epilog="참고: 전체 데이터 수집은 collect_all_data.py를 사용하세요."
    )
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="수집할 회차 (예: 제418회)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API 인증키 (지정하지 않으면 ASSEMBLY_API_KEY 환경 변수 사용)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="테스트용 최대 수집 개수"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력 (디버깅용)"
    )
    
    args = parser.parse_args()
    
    # API 키 처리: CLI 인자 > 환경 변수 > 에러
    api_key = args.api_key or os.getenv("ASSEMBLY_API_KEY")
    if not api_key:
        print("❌ API 인증키가 필요합니다.")
        print("   --api-key 옵션을 사용하거나 .env 파일에 ASSEMBLY_API_KEY를 설정해주세요.")
        sys.exit(1)
    
    collect_session_with_html(args.session, api_key, args.limit, verbose=args.verbose)

