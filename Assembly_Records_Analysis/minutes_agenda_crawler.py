# -*- coding: utf-8 -*-
"""
국회회의록(웹 뷰어) 상세 페이지 크롤링 스크립트 (디버그/항상-저장 + 안건매핑)
- meeting_no 추출 강화, data_mem_id 안전 추출
- place/open_time/page_url 제거
- minutes_speeches.csv 에 안건 필드(agenda_item_orders, agenda_item_titles, agenda_item_times) 추가
- 안건바가 연달아 2개 이상 나오는 경우, '발언자 등장 전까지' 연속된 안건들을 하나의 그룹으로 묶어 합쳐서 매핑

pip install requests beautifulsoup4 pandas
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MOIS-Intern-Project/1.0)"}
DEBUG   = True
RETRY   = 2
SLEEP_SEC = 0.7

# 크롤링할 '회의록 뷰어 상세 URL'
DETAIL_URLS = [
    "https://record.assembly.go.kr/assembly/viewer/minutes/xml.do?id=55384&type=view"
    
]

# ----------------- 유틸 -----------------
def log(*args):
    if DEBUG:
        print(*args)

def clean(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def get_soup(url: str) -> BeautifulSoup:
    last_err = None
    for i in range(RETRY):
        try:
            log(f"[GET] {url}  (try {i+1}/{RETRY})")
            r = requests.get(url, headers=HEADERS, timeout=25)
            log(f"  -> status {r.status_code}, len={len(r.text)}")
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            if not soup.select_one("#minutes"):
                log("  !! '#minutes' 섹션을 찾지 못했습니다.")
            return soup
        except Exception as e:
            last_err = e
            log(f"  !! request error: {e}")
            time.sleep(0.8)
    raise last_err

# ----------------- 상단 요약 -----------------
def parse_header_title(soup: BeautifulSoup):
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
            m3 = re.search(r"제\s*\d+\s*호", header_text)
            if m3: meeting_no = m3.group(0)
    log(f"  header: session={session}, session_type={session_type}, meeting_no={meeting_no}")
    return session, session_type, meeting_no

def parse_date_only(soup: BeautifulSoup):
    date_text = ""
    for li in soup.select("#minutes .minutes_header .place ul > li"):
        sbj_el = li.select_one(".sbj, .sbj.lts2, .sbj.lts4")
        sbj = clean(sbj_el.get_text()) if sbj_el else ""
        if "일시" in sbj:
            con_el = li.select_one("p.con")
            if con_el:
                date_text = clean(con_el.get_text())
            break
    log(f"  date: {date_text}")
    return date_text

def parse_header(url: str):
    soup = get_soup(url)
    session, session_type, meeting_no = parse_header_title(soup)
    date_text = parse_date_only(soup)

    rows = []
    # 일시 1행
    rows.append({
        "session": session, "session_type": session_type, "meeting_no": meeting_no,
        "date": date_text, "section": "일시", "item_order": "", "item_text": date_text,
    })

    # 의사일정/상정된 안건
    cnt_agenda = cnt_sched = 0
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
                if label == "의사일정": cnt_agenda += 1
                else: cnt_sched += 1
    log(f"  agenda_items={cnt_agenda}, scheduled_items={cnt_sched}")
    return rows

# ----------------- 본문: 안건바 → 발언 매핑 -----------------
def get_text_with_br(container: Tag) -> str:
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
    """a.tit 텍스트(권장) 또는 p 안의 텍스트에서 'n. 제목'을 분리"""
    txt = ""
    if a_or_p:
        if a_or_p.name == "a":
            txt = clean(a_or_p.get_text())
        else:
            txt = clean(a_or_p.get_text())
    m = re.match(r"(\d+)\.\s*(.+)", txt)
    if m:
        return m.group(1), clean(m.group(2))   # (번호, 제목)
    return "", txt

def _nearest_time_after(node: Tag):
    """현재 안건바 뒤쪽에서 가까운 p.tit_sm.taR 을 찾아 시각 텍스트를 돌려줌(없으면 빈 문자열)"""
    cur = node
    for _ in range(12):  # 너무 멀리 가지 않도록 제한
        cur = cur.find_next()
        if not isinstance(cur, Tag):
            continue
        cls = cur.get("class", [])
        if cur.name == "p" and "tit_sm" in cls and "taR" in cls:
            return clean(cur.get_text())
        if cur.name == "p" and "tit_sm" in cls and "angun" in cls:
            # 다음 안건바가 바로 나오면 종료
            break
        if cur.name == "div" and "speaker" in cls:
            break
    return ""

def _extract_agenda_groups(soup: BeautifulSoup):
    """
    minutes_body에서 안건바들을 '발언자 등장 전까지 연속된 것'끼리 묶어 그룹화.
    각 그룹 = {'start_order': int, 'orders': '1,2', 'titles': 'A / B', 'times': '(15시..) / (15시..)' }
    """
    speakers = soup.select("#minutes .minutes_body .speaker")
    bars = soup.select("#minutes .minutes_body p.tit_sm.angun")
    log(f"  agenda bars: {len(bars)}, speakers: {len(speakers)}")

    groups = []
    i = 0
    while i < len(bars):
        cur_bar = bars[i]
        orders, titles, times = [], [], []

        # 현재 바부터 시작해서 '다음 발언자 전까지' 등장하는 바들을 한 그룹으로 묶음
        while True:
            a_tit = cur_bar.select_one("a.tit")
            num, title = _parse_agenda_title(a_tit if a_tit else cur_bar)
            if num: orders.append(num)
            if title: titles.append(title)
            tm = _nearest_time_after(cur_bar)
            if tm: times.append(tm)

            # 다음 바와 현재 바 사이에 speaker가 있는지 검사 → 있으면 그룹 종료
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
            # speaker가 없고 바로 다음도 바라면 같은 그룹에 포함
            i += 1
            cur_bar = next_bar

        # 그룹의 시작 발언 순번: 그룹의 마지막 바 이후 등장하는 첫 speaker
        first_spk_after_group = cur_bar.find_next(class_="speaker")
        start_order = None
        if first_spk_after_group:
            for idx, spk in enumerate(speakers, start=1):
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
    log("  agenda groups (start_order → orders | titles):")
    for g in groups:
        log(f"    {g['start_order']:>3} → {g['orders']} | {g['titles']}")
    return groups, speakers

def parse_speeches(url: str):
    soup = get_soup(url)
    session, session_type, meeting_no = parse_header_title(soup)
    date_text = parse_date_only(soup)

    # 안건 그룹 추출
    groups, speakers = _extract_agenda_groups(soup)

    # 구간화: 각 그룹은 [start, next_start-1] 범위를 가짐
    ranges = []
    for idx, g in enumerate(groups):
        start = g["start_order"]
        end = (groups[idx+1]["start_order"] - 1) if idx+1 < len(groups) else len(speakers)
        ranges.append((start, end, g))

    rows = []
    for order, spk in enumerate(speakers, start=1):
        # data_mem_id 안정 추출
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

        # 이 발언이 속한 안건 그룹 찾기
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
            "agenda_item_orders": ag_orders,   # 새 필드
            "agenda_item_titles": ag_titles,   # 새 필드
            "agenda_item_times": ag_times,     # 새 필드
            "speech_text": speech,
        })
    return rows

# ----------------- 실행 -----------------
def run(urls,
        out_header_csv="minutes_header_summary.csv",
        out_speech_csv="minutes_speeches.csv"):
    header_all, speech_all = [], []

    if not urls:
        print("DETAIL_URLS 가 비어 있습니다. URL을 1개 이상 추가하세요.")
        pd.DataFrame(columns=[
            "session","session_type","meeting_no","date","section","item_order","item_text"
        ]).to_csv(out_header_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=[
            "session","session_type","meeting_no","date","speech_order",
            "speaker_name","speaker_position","speaker_area","data_mem_id",
            "agenda_item_orders","agenda_item_titles","agenda_item_times","speech_text"
        ]).to_csv(out_speech_csv, index=False, encoding="utf-8-sig")
        print(f"빈 CSV를 생성했습니다: {out_header_csv}, {out_speech_csv}")
        return

    for u in urls:
        try:
            log(f"\n=== Parse header: {u}")
            header_all.extend(parse_header(u))
            time.sleep(SLEEP_SEC)

            log(f"=== Parse speeches(+agenda): {u}")
            speech_all.extend(parse_speeches(u))
            time.sleep(SLEEP_SEC)
        except Exception as e:
            print("[ERROR]", u, e)

    # 항상 CSV 생성
    pd.DataFrame(header_all, columns=[
        "session","session_type","meeting_no","date","section","item_order","item_text"
    ]).to_csv(out_header_csv, index=False, encoding="utf-8-sig")

    pd.DataFrame(speech_all, columns=[
        "session","session_type","meeting_no","date","speech_order",
        "speaker_name","speaker_position","speaker_area","data_mem_id",
        "agenda_item_orders","agenda_item_titles","agenda_item_times","speech_text"
    ]).to_csv(out_speech_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {out_header_csv} (rows={len(header_all)})")
    print(f"Saved: {out_speech_csv} (rows={len(speech_all)})")

if __name__ == "__main__":
    run(DETAIL_URLS)
