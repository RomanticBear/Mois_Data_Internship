# -*- coding: utf-8 -*-
"""
국회회의록(웹 뷰어) 상세 페이지 크롤링 스크립트 (다중 id + 회차별 폴더 저장 + 안건매핑)
- 여러 minutes id를 순회하며 파싱
- session(제 n회) 폴더 생성 후, meeting_no(제n차/제n호)별로 2개 CSV 저장
  └ {base_out}/제{session}회/{meeting_no}_minutes_header_summary.csv
  └ {base_out}/제{session}회/{meeting_no}_minutes_speeches.csv
- minutes_speeches.csv 에 안건 필드(agenda_item_orders, agenda_item_titles, agenda_item_times) 포함
"""

import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MOIS-Intern-Project/1.0)"}
DEBUG   = True
RETRY   = 2
SLEEP_SEC = 0.6

# ==========================
# 1) 여기에 minutes id 목록 입력
# ==========================
MINUTES_IDS = [
    52081, 52082, 52083, 52130, 52131, 52132, 52133, 52134, 52135, 52409, 52410,
    52411, 52412, 52413, 52414, 52415, 52416, 52417, 52418, 52419, 52420, 52421,
    52422, 52615, 52649, 52650, 52687, 54447, 52726, 52727, 52770, 54593, 54615,
    54650, 54950, 54947, 55011, 55071, 55130, 55234, 55266, 55338, 55375, 55379,
    55395, 55396, 55410, 55454, 55680, 55662
]

# DETAIL_URLS 자동 생성
DETAIL_URLS = [f"https://record.assembly.go.kr/assembly/viewer/minutes/xml.do?id={i}&type=view"
               for i in MINUTES_IDS]

# 저장 루트 폴더(원하면 바꿔도 OK)
BASE_OUTDIR = "minutes_output"

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
            m3 = re.search(r"제\s*\d+\s*[차호]", header_text)
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
    return rows, (session, meeting_no)

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
    txt = ""
    if a_or_p:
        txt = clean(a_or_p.get_text())
    m = re.match(r"(\d+)\.\s*(.+)", txt)
    if m:
        return m.group(1), clean(m.group(2))
    return "", txt

def _nearest_time_after(node: Tag):
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
    speakers = soup.select("#minutes .minutes_body .speaker")
    bars = soup.select("#minutes .minutes_body p.tit_sm.angun")
    log(f"  agenda bars: {len(bars)}, speakers: {len(speakers)}")

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
    log("  agenda groups (start_order → orders | titles):")
    for g in groups:
        log(f"    {g['start_order']:>3} → {g['orders']} | {g['titles']}")
    return groups, soup.select("#minutes .minutes_body .speaker")

def parse_speeches(url: str):
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

# ----------------- 저장 헬퍼 -----------------
def safe_name(s: str) -> str:
    """파일/폴더명 안전화"""
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    s = s.replace("?", "").replace("*", "").replace('"', "").replace("<", "").replace(">", "").replace("|", "")
    return s.strip()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ----------------- 실행 -----------------
def run_all(urls, base_outdir=BASE_OUTDIR):
    if not urls:
        print("DETAIL_URLS 가 비어 있습니다.")
        return

    total_header, total_speech = 0, 0

    for u in urls:
        print(f"\n==== Parse: {u} ====")
        # 1) 헤더
        header_rows, key1 = [], (None, "")
        try:
            h_rows, key1 = parse_header(u)
            header_rows.extend(h_rows)
        except Exception as e:
            print("  [ERROR] header:", e)

        # 2) 본문(발언+안건)
        speech_rows, key2 = [], (None, "")
        try:
            s_rows, key2 = parse_speeches(u)
            speech_rows.extend(s_rows)
        except Exception as e:
            print("  [ERROR] speeches:", e)

        # session / meeting_no 결정 (둘 중 하나라도 성공한 쪽 기준)
        session = key1[0] if key1[0] is not None else key2[0]
        meeting_no = key1[1] if key1[1] else key2[1]

        # session/meeting_no 없으면 id로 폴백
        if session is None: session = "unknown"
        if not meeting_no:  meeting_no = "unknown"

        # 회차 폴더 및 파일 경로
        session_dir = os.path.join(base_outdir, f"제{session}회")
        ensure_dir(session_dir)

        fname_prefix = safe_name(meeting_no)  # 예: "제1차" 또는 "제1호"
        path_header  = os.path.join(session_dir, f"{fname_prefix}_minutes_header_summary.csv")
        path_speech  = os.path.join(session_dir, f"{fname_prefix}_minutes_speeches.csv")

        # 저장 (항상 파일 생성)
        pd.DataFrame(header_rows, columns=[
            "session","session_type","meeting_no","date","section","item_order","item_text"
        ]).to_csv(path_header, index=False, encoding="utf-8-sig")

        pd.DataFrame(speech_rows, columns=[
            "session","session_type","meeting_no","date","speech_order",
            "speaker_name","speaker_position","speaker_area","data_mem_id",
            "agenda_item_orders","agenda_item_titles","agenda_item_times","speech_text"
        ]).to_csv(path_speech, index=False, encoding="utf-8-sig")

        print(f"  -> Saved: {path_header} (rows={len(header_rows)})")
        print(f"  -> Saved: {path_speech} (rows={len(speech_rows)})")
        total_header += len(header_rows)
        total_speech += len(speech_rows)
        time.sleep(SLEEP_SEC)

    print(f"\n[완료] 총 헤더행 {total_header} / 발언행 {total_speech} 저장, 폴더 루트 = {os.path.abspath(base_outdir)}")

if __name__ == "__main__":
    run_all(DETAIL_URLS)
