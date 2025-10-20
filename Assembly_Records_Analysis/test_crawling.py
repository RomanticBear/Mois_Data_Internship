"""
국회회의록(웹 뷰어) 상세 페이지 크롤링 스크립트
- 상단 요약(회차/일시/장소/의사일정/상정된 안건) -> minutes_header_summary.csv
- 발언자/발언내용(발화 단위) -> minutes_speeches.csv

필요 라이브러리: requests, bs4, pandas
pip install requests beautifulsoup4 pandas
"""

import re
import time
import csv
import requests
import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.parse import urljoin

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MOIS-Intern-Project/1.0)"
}

# (예시) 여기 리스트에 '회의록뷰어 상세 URL'을 넣어 실행하세요.
DETAIL_URLS = [
    "https://record.assembly.go.kr/assembly/viewer/minutes/xml.do?id=52130&type=view"
]

# ========== 공통 유틸 ==========
def clean(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\xa0", " ")    # &nbsp;
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def get_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

# ========== 상단 요약 영역 파싱 ==========
def parse_header_title(soup: BeautifulSoup):
    """#minutes .minutes_header .tit_wrap 안의 p.turn / div.tit_in p.num"""
    session = None
    session_type = ""
    meeting_no = ""

    tw = soup.select_one("#minutes .minutes_header .tit_wrap")
    if tw:
        turn = tw.select_one("p.turn")
        if turn:
            t = turn.get_text("\n", strip=True)
            m1 = re.search(r"제\s*(\d+)\s*회", t)
            if m1:
                session = int(m1.group(1))
            m2 = re.search(r"\(([^)]+)\)", t)
            if m2:
                session_type = clean(m2.group(1))
        pnum = tw.select_one("div.tit_in p.num")
        if pnum:
            meeting_no = clean(pnum.get_text())
    return session, session_type, meeting_no

def parse_place_block(soup: BeautifulSoup):
    """
    #minutes .minutes_header .place > ul > li
      - .sbj(.lts2/.lts4) : 라벨(일시/장소/의사일정/상정된 안건)
      - p.con : 값(일시/장소)
      - ul.list_num li : 항목(의사일정/상정된 안건)
    """
    date_text = ""
    place_text = ""
    agenda_items = []
    scheduled_items = []

    for li in soup.select("#minutes .minutes_header .place ul > li"):
        sbj_el = li.select_one(".sbj, .sbj.lts2, .sbj.lts4")
        sbj = clean(sbj_el.get_text()) if sbj_el else ""
        con_el = li.select_one("p.con")
        list_els = li.select("ul.list_num li")

        if "일시" in sbj:
            if con_el:
                date_text = clean(con_el.get_text())
        elif "장소" in sbj:
            if con_el:
                place_text = clean(con_el.get_text())
        elif "의사일정" in sbj:
            for idx, it in enumerate(list_els, start=1):
                a = it.select_one("a")
                txt = clean(a.get_text()) if a and clean(a.get_text()) else clean(it.get_text())
                txt = re.sub(r"^\d+\.\s*", "", txt)
                agenda_items.append((idx, txt))
        elif "상정된 안건" in sbj:
            for idx, it in enumerate(list_els, start=1):
                a = it.select_one("a")
                txt = clean(a.get_text()) if a and clean(a.get_text()) else clean(it.get_text())
                txt = re.sub(r"^\d+\.\s*", "", txt)
                scheduled_items.append((idx, txt))

    return date_text, place_text, agenda_items, scheduled_items

def parse_header(url: str):
    soup = get_soup(url)
    session, session_type, meeting_no = parse_header_title(soup)
    date_text, place_text, agenda_items, scheduled_items = parse_place_block(soup)

    rows = []
    # 일시/장소
    rows.append({
        "session": session, "session_type": session_type, "meeting_no": meeting_no,
        "date": date_text, "place": place_text, "section": "일시",
        "item_order": "", "item_text": date_text, "page_url": url
    })
    rows.append({
        "session": session, "session_type": session_type, "meeting_no": meeting_no,
        "date": date_text, "place": place_text, "section": "장소",
        "item_order": "", "item_text": place_text, "page_url": url
    })
    # 의사일정
    for order, txt in agenda_items:
        rows.append({
            "session": session, "session_type": session_type, "meeting_no": meeting_no,
            "date": date_text, "place": place_text, "section": "의사일정",
            "item_order": order, "item_text": txt, "page_url": url
        })
    # 상정된 안건
    for order, txt in scheduled_items:
        rows.append({
            "session": session, "session_type": session_type, "meeting_no": meeting_no,
            "date": date_text, "place": place_text, "section": "상정된 안건",
            "item_order": order, "item_text": txt, "page_url": url
        })
    return rows

# ========== 발언(발화) 파싱 ==========
def get_text_with_br(container: Tag) -> str:
    """
    .talk .txt 안에 span.spk_sub, <br> 등이 섞여 있음.
    - 줄바꿈은 공백 하나로 정규화
    - 여는/닫는 공백 정리
    """
    if container is None:
        return ""

    parts = []
    for node in container.descendants:
        if isinstance(node, NavigableString):
            parts.append(str(node))
        elif isinstance(node, Tag) and node.name.lower() == "br":
            parts.append("\n")
    text = "".join(parts)
    # span마다 &nbsp; + 줄바꿈 많으므로 정리
    text = text.replace("\r", "\n").replace("\xa0", " ")
    # 연속 개행을 한 개행으로
    text = re.sub(r"\n\s*\n+", "\n", text)
    # 줄 끝 공백 제거 후 한 줄로 뽑고 싶으면 아래 주석 해제
    # text = text.replace("\n", " ")
    # 다만 발언문은 문단 구분이 유의미할 수 있어 개행 유지
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def parse_speeches(url: str):
    """
    #minutes .minutes_body 내부의 .speaker 블록들을 순회
    - data-attrs: data-mem_id, data-name, data-pos (있으면 활용)
    - 화면 표기: 직위(.position), 이름(strong.name), 지역/소속(.area)
    - 본문: .talk .txt
    """
    soup = get_soup(url)

    # 상단 공통 메타
    session, session_type, meeting_no = parse_header_title(soup)
    date_text, place_text, _, _ = parse_place_block(soup)
    open_time = clean(soup.select_one("#minutes .minutes_body > p.tit_sm.taR").get_text()) \
        if soup.select_one("#minutes .minutes_body > p.tit_sm.taR") else ""

    rows = []
    order = 0
    for spk in soup.select("#minutes .minutes_body .speaker"):
        order += 1
        # data-* 속성(있을 때만)
        data_mem_id = spk.get("data-mem_id", "")
        data_name_attr = spk.get("data-name", "")
        data_pos_attr = spk.get("data-pos", "")

        # 상단 프로필
        name = clean((spk.select_one(".man .txt strong.name") or {}).get_text() if spk.select_one(".man .txt strong.name") else data_name_attr)
        position = clean((spk.select_one(".man .txt .position") or {}).get_text() if spk.select_one(".man .txt .position") else data_pos_attr)
        area = clean((spk.select_one(".man .txt .area") or {}).get_text() if spk.select_one(".man .txt .area") else "")

        # 발언 본문
        text_container = spk.select_one(".talk .txt")
        speech = get_text_with_br(text_container)

        rows.append({
            "session": session,
            "session_type": session_type,
            "meeting_no": meeting_no,
            "date": date_text,
            "place": place_text,
            "open_time": open_time,          # 예: (10시03분 개의)
            "speech_order": order,
            "speaker_name": name,
            "speaker_position": position,    # 예: 위원장/간사/위원/기관직함 등
            "speaker_area": area,            # 예: (전남 나주시화순군)
            "data_mem_id": data_mem_id,      # 페이지 내 고유 식별자 힌트
            "speech_text": speech,
            "page_url": url
        })
    return rows

# ========== 실행 진입점 ==========
def run(urls,
        out_header_csv="minutes_header_summary.csv",
        out_speech_csv="minutes_speeches.csv",
        sleep_sec=0.7):
    header_all, speech_all = [], []
    for u in urls:
        try:
            header_all.extend(parse_header(u))
            speech_all.extend(parse_speeches(u))
        except Exception as e:
            print("[ERROR]", u, e)
        time.sleep(sleep_sec)

    # 저장
    if header_all:
        pd.DataFrame(header_all, columns=[
            "session","session_type","meeting_no","date","place",
            "section","item_order","item_text","page_url"
        ]).to_csv(out_header_csv, index=False, encoding="utf-8-sig")
        print(f"Saved header summary -> {out_header_csv} (rows={len(header_all)})")

    if speech_all:
        pd.DataFrame(speech_all, columns=[
            "session","session_type","meeting_no","date","place","open_time",
            "speech_order","speaker_name","speaker_position","speaker_area",
            "data_mem_id","speech_text","page_url"
        ]).to_csv(out_speech_csv, index=False, encoding="utf-8-sig")
        print(f"Saved speeches -> {out_speech_csv} (rows={len(speech_all)})")

if __name__ == "__main__":
    run(DETAIL_URLS)
