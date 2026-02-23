"""
회의록 API를 통해 PDF 파일을 다운로드하고 회차별로 저장하는 스크립트
회의록API_PDF다운.ipynb 참고
"""
import os
import sys
import math
import time
import re
from pathlib import Path
from urllib.parse import unquote
from dotenv import load_dotenv
import requests
import pandas as pd
import urllib3
from tqdm import tqdm

# SSL 인증 경고 숨김
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "backend"))

load_dotenv()

# ----- 설정값 -----
BASE_URL = "https://open.assembly.go.kr/portal/openapi/ncwgseseafwbuheph"
API_KEY = os.getenv("NA_OPEN_API_KEY", "065cc2aaf8fc41489588360aae7b9270")
P_SIZE = 100
TIMEOUT = 20
RETRY = 3
ASSEMBLY_NUMBER = os.getenv("ASSEMBLY_NUMBER", "21")
TARGET_COMMITTEE = os.getenv("TARGET_COMMITTEE", "행정안전위원회")
TARGET_YEARS = [y.strip() for y in os.getenv("TARGET_YEARS", "").split(",") if y.strip()]
EXCLUDE_SUBCOMMITTEE = os.getenv("EXCLUDE_SUBCOMMITTEE", "true").lower() in ("1", "true", "yes", "y")

# 공통 파라미터 설정
params_common = {
    "KEY": API_KEY,
    "Type": "json",
    "pIndex": 1,
    "pSize": P_SIZE,
    "DAE_NUM": ASSEMBLY_NUMBER,
    "COMM_NAME": TARGET_COMMITTEE,
}


def _extract_head_and_rows(js):
    """JSON 구조 파서"""
    root = js.get("ncwgseseafwbuheph")
    if root is None:
        return None, []

    if isinstance(root, dict):
        head = None
        if "head" in root:
            head = root["head"][0] if isinstance(root["head"], list) else root["head"]
        rows = root.get("row", [])
    else:
        head, rows = None, []
        for item in root:
            if not isinstance(item, dict):
                continue
            if "head" in item and head is None:
                head = item["head"][0] if isinstance(item["head"], list) else item["head"]
            if "row" in item and not rows:
                rows = item["row"]

    if isinstance(rows, dict):
        rows = [rows]
    return head, rows


def get_with_retry(url, params, timeout=TIMEOUT, retry=RETRY):
    """네트워크 요청 헬퍼 함수"""
    last_err = None
    for attempt in range(1, retry + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, verify=False)
            if resp.status_code == 200:
                return resp
            last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            last_err = e
        time.sleep(0.8 * attempt)
    raise last_err


def _parse_result(head):
    """RESULT 파싱"""
    result = head.get("RESULT", None)
    code, msg = None, None

    if isinstance(result, dict):
        code = result.get("CODE") or result.get("code")
        msg = result.get("MESSAGE") or result.get("message")
    elif isinstance(result, list):
        if result:
            first = result[0]
            if isinstance(first, dict):
                code = first.get("CODE") or first.get("code")
                msg = first.get("MESSAGE") or first.get("message")
            elif isinstance(first, str):
                code = first
    elif isinstance(result, str):
        code = result

    return code, msg


def fetch_total_count(filters=None):
    """전체 데이터 개수 조회"""
    params = params_common.copy()
    if filters:
        params.update(filters)
    params["pIndex"] = 1
    params["pSize"] = 1

    r = get_with_retry(BASE_URL, params)
    js = r.json()

    head, _ = _extract_head_and_rows(js)
    if not head:
        raise RuntimeError(f"API 응답 구조 파싱 실패: {str(js)[:300]}")

    code, msg = _parse_result(head)

    total_raw = head.get("list_total_count", 0)
    try:
        total = int(total_raw)
    except Exception:
        total = 0

    if code not in (None, "INFO-000") and total == 0:
        raise RuntimeError(f"API 에러: CODE={code}, MESSAGE={msg}")

    return total


def fetch_page(p_index=1, p_size=P_SIZE, filters=None):
    """단일 페이지 데이터 조회"""
    params = params_common.copy()
    if filters:
        params.update(filters)
    params["pIndex"] = p_index
    params["pSize"] = p_size

    r = get_with_retry(BASE_URL, params)
    js = r.json()

    _, rows = _extract_head_and_rows(js)
    return rows


def fetch_all_rows(filters=None):
    """전체 페이지 순회 수집"""
    total = fetch_total_count(filters=filters)
    if total == 0:
        return []

    pages = math.ceil(total / P_SIZE)
    all_rows = []

    for p in range(1, pages + 1):
        rows = fetch_page(p_index=p, p_size=P_SIZE, filters=filters)
        all_rows.extend(rows)
        time.sleep(0.5)  # API 호출 간격
    
    return all_rows


def extract_session_number(filename):
    """파일명에서 회차 번호 추출 (예: 제430회 -> 430)"""
    # 패턴: 제XXX회
    match = re.search(r'제(\d+)회', filename)
    if match:
        return match.group(1)
    return None


def get_session_folder_name(session_num):
    """회차 번호로 폴더명 생성 (예: 430 -> 제430회)"""
    if session_num:
        return f"제{session_num}회"
    return "기타"


def is_subcommittee_row(row):
    """소위원회 관련 회의록 여부 판별"""
    text_fields = [
        row.get("CLASS_NAME", ""),
        row.get("SUB_NAME", ""),
        row.get("TITLE", ""),
        row.get("CONF_ID", ""),
    ]
    joined = " ".join(str(v) for v in text_fields if pd.notna(v))
    return "소위원회" in joined


def get_default_years_for_assembly(assembly_number):
    """대수별 기본 연도 범위 반환 (API 연도 필수 대응)"""
    if assembly_number == "20":
        return ["2020", "2019", "2018", "2017", "2016"]
    if assembly_number == "21":
        return ["2024", "2023", "2022", "2021", "2020"]
    if assembly_number == "22":
        return ["2026", "2025", "2024"]
    return ["2026", "2025", "2024", "2023", "2022", "2021", "2020"]


def download_pdf(url, save_dir, session_folder):
    """PDF 파일 다운로드"""
    try:
        r = requests.get(url, verify=False, timeout=30)
        r.raise_for_status()

        # 파일명 추출 + 디코딩
        cd = r.headers.get("Content-Disposition", "")
        if not cd:
            # Content-Disposition이 없으면 URL에서 파일명 추출 시도
            filename = f"document_{int(time.time())}.pdf"
        else:
            raw = cd.split("filename=")[-1].strip('"; ')
            filename = unquote(raw)

        # 금지문자 제거
        for ch in '\\/:*?"<>|':
            filename = filename.replace(ch, "_")

        # 회차별 폴더 생성
        session_dir = save_dir / session_folder
        session_dir.mkdir(parents=True, exist_ok=True)

        # 저장 경로
        save_path = session_dir / filename

        # 이미 있으면 건너뜀
        if save_path.exists():
            return save_path, True  # (경로, 스킵 여부)

        # 저장
        with open(save_path, "wb") as f:
            f.write(r.content)

        return save_path, False  # (경로, 스킵 여부)

    except Exception as e:
        raise Exception(f"다운로드 실패: {str(e)}")


def main():
    """메인 함수"""
    print("=" * 80)
    print("회의록 PDF 다운로드 및 회차별 저장")
    print("=" * 80)
    print()

    # 저장 디렉토리 설정
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 저장 디렉토리: {data_dir}")
    print()

    # 1. API로 회의록 데이터 수집
    print("📡 API로 회의록 데이터 수집 중...")
    print(f"  - 대상 대수: {ASSEMBLY_NUMBER}대")
    print(f"  - 대상 위원회: {TARGET_COMMITTEE}")
    print(f"  - 소위원회 제외: {'예' if EXCLUDE_SUBCOMMITTEE else '아니오'}")
    print()

    all_rows = []

    effective_years = TARGET_YEARS[:] if TARGET_YEARS else get_default_years_for_assembly(ASSEMBLY_NUMBER)

    if effective_years:
        if TARGET_YEARS:
            print(f"  - 대상 연도: {', '.join(effective_years)}")
        else:
            print(f"  - 대상 연도: {', '.join(effective_years)} (기본값)")
        for year in effective_years:
            print(f"    · {year}년 데이터 수집 중...")
            try:
                rows = fetch_all_rows(filters={"CONF_DATE": year})
                all_rows.extend(rows)
                print(f"      ✅ {len(rows)}건 수집 완료")
            except RuntimeError as e:
                if "해당하는 데이터가 없습니다" in str(e) or "INFO-200" in str(e):
                    print(f"      ⚠️ {year}년 데이터가 없습니다. 스킵합니다.")
                else:
                    print(f"      ❌ {year}년 데이터 수집 실패: {e}")
                    raise
    else:
        print("  - 대상 연도: 전체 (연도 필터 없음)")
        try:
            all_rows = fetch_all_rows()
            print(f"    ✅ {len(all_rows)}건 수집 완료")
        except RuntimeError as e:
            print(f"    ❌ 전체 데이터 수집 실패: {e}")
            raise
    
    if not all_rows:
        print("❌ 수집된 데이터가 없습니다.")
        return

    print(f"\n✅ 총 {len(all_rows)}건 수집 완료")
    print()

    # 2. DataFrame 구성
    print("📊 데이터 정리 중...")
    df = pd.DataFrame(all_rows)
    
    keep_cols = [
        "CONFER_NUM", "CONF_ID", "TITLE", "CLASS_NAME", "DAE_NUM",
        "COMM_NAME", "CONF_DATE", "SUB_NAME",
        "VOD_LINK_URL", "CONF_LINK_URL", "PDF_LINK_URL", "PDF_FILE_ID", "DEPT_CD"
    ]
    cols = [c for c in keep_cols if c in df.columns]
    df = df[cols].copy()

    # PDF 링크가 있는 데이터만 필터링
    df_pdfs = df[df["PDF_LINK_URL"].notna()].copy()

    # 소위원회 제외 옵션 적용
    if EXCLUDE_SUBCOMMITTEE and not df_pdfs.empty:
        before_count = len(df_pdfs)
        df_pdfs = df_pdfs[~df_pdfs.apply(is_subcommittee_row, axis=1)].copy()
        excluded_count = before_count - len(df_pdfs)
        print(f"🚫 소위원회 제외: {excluded_count}건")

    if df_pdfs.empty:
        print("❌ PDF 링크가 있는 데이터가 없습니다.")
        return

    print(f"✅ PDF 링크가 있는 데이터: {len(df_pdfs)}건")
    print()

    # 3. 고유 PDF 링크 추출
    unique_pdfs = df_pdfs["PDF_LINK_URL"].drop_duplicates()
    print(f"📋 고유 PDF 링크: {len(unique_pdfs)}개")
    print()

    # 4. PDF 다운로드
    print("📥 PDF 다운로드 시작...")
    print()

    results = {
        "success": [],
        "skipped": [],
        "failed": []
    }

    for idx, url in enumerate(tqdm(unique_pdfs, desc="PDF 다운로드"), 1):
        try:
            # 해당 URL의 데이터에서 회차 정보 추출 시도
            row = df_pdfs[df_pdfs["PDF_LINK_URL"] == url].iloc[0]
            
            # TITLE에서 회차 추출 시도
            session_num = None
            if "TITLE" in row and pd.notna(row["TITLE"]):
                session_num = extract_session_number(str(row["TITLE"]))
            
            # CONF_ID나 다른 필드에서도 추출 시도
            if not session_num and "CONF_ID" in row and pd.notna(row["CONF_ID"]):
                session_num = extract_session_number(str(row["CONF_ID"]))
            
            session_folder = get_session_folder_name(session_num)
            
            save_path, skipped = download_pdf(url, data_dir, session_folder)
            
            if skipped:
                results["skipped"].append({
                    "url": url,
                    "path": str(save_path),
                    "session": session_folder
                })
            else:
                results["success"].append({
                    "url": url,
                    "path": str(save_path),
                    "session": session_folder
                })
            
            # API 호출 간격
            time.sleep(0.5)
            
        except Exception as e:
            results["failed"].append({
                "url": url,
                "error": str(e)
            })
            print(f"\n❌ 다운로드 실패: {url}")
            print(f"   오류: {e}")

    # 5. 결과 요약
    print("\n" + "=" * 80)
    print("📊 다운로드 결과 요약")
    print("=" * 80)
    print(f"  ✅ 성공: {len(results['success'])}개")
    print(f"  ⏭️  스킵 (이미 존재): {len(results['skipped'])}개")
    print(f"  ❌ 실패: {len(results['failed'])}개")
    print()

    # 회차별 통계
    session_stats = {}
    for item in results["success"] + results["skipped"]:
        session = item["session"]
        session_stats[session] = session_stats.get(session, 0) + 1

    if session_stats:
        print("📁 회차별 저장 통계:")
        for session, count in sorted(session_stats.items()):
            print(f"  - {session}: {count}개 파일")

    if results["failed"]:
        print("\n❌ 실패한 파일:")
        for failed in results["failed"]:
            print(f"  - {failed['url']}: {failed['error']}")

    print("\n" + "=" * 80)
    print("✅ 작업 완료!")
    print("=" * 80)
    print(f"\n📁 저장 위치: {data_dir}")


if __name__ == "__main__":
    main()

