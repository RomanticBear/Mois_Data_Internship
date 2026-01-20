# -*- coding: utf-8 -*-
"""
국회 위원회 회의록 Open API 클라이언트
- 위원회 회의록 API를 통한 데이터 수집
- 페이징 처리 및 자동화 지원
"""

import time
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CommitteeMeetingRecord:
    """위원회 회의록 레코드"""
    confer_num: str  # 회의번호
    title: str  # 회의명
    class_name: str  # 회의종류명
    dae_num: str  # 대수
    comm_name: str  # 위원회명
    vodcomm_code: str  # 영상회의록
    conf_date: str  # 회의날짜
    sub_name: str  # 안건명
    vod_link_url: str  # 영상회의록 링크
    conf_link_url: str  # 요약정보 팝업
    pdf_link_url: str  # PDF파일 링크
    pdf_file_id: str  # 회의록
    dept_cd: str  # 위원회코드
    conf_id: str  # 회의ID
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommitteeMeetingRecord":
        """딕셔너리에서 객체 생성"""
        return cls(
            confer_num=data.get("CONFER_NUM", ""),
            title=data.get("TITLE", ""),
            class_name=data.get("CLASS_NAME", ""),
            dae_num=data.get("DAE_NUM", ""),
            comm_name=data.get("COMM_NAME", ""),
            vodcomm_code=data.get("VODCOMM_CODE", ""),
            conf_date=data.get("CONF_DATE", ""),
            sub_name=data.get("SUB_NAME", ""),
            vod_link_url=data.get("VOD_LINK_URL", ""),
            conf_link_url=data.get("CONF_LINK_URL", ""),
            pdf_link_url=data.get("PDF_LINK_URL", ""),
            pdf_file_id=data.get("PDF_FILE_ID", ""),
            dept_cd=data.get("DEPT_CD", ""),
            conf_id=data.get("CONF_ID", ""),
        )


class AssemblyAPIClient:
    """국회 Open API 클라이언트"""
    
    BASE_URL = "https://open.assembly.go.kr/portal/openapi/ncwgseseafwbuheph"
    
    def __init__(
        self,
        api_key: str,
        response_type: str = "json",
        page_size: int = 100,
        request_delay: float = 0.5
    ):
        """
        Args:
            api_key: API 인증키
            response_type: 응답 형식 (json, xml)
            page_size: 페이지당 요청 숫자
            request_delay: 요청 간 지연 시간 (초)
        """
        self.api_key = api_key
        self.response_type = response_type
        self.page_size = page_size
        self.request_delay = request_delay
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MOIS-Intern-Project/1.0)"
        }
    
    def _make_request(
        self,
        params: Dict[str, Any],
        page_index: int = 1
    ) -> Dict[str, Any]:
        """API 요청 실행"""
        request_params = {
            "KEY": self.api_key,
            "Type": self.response_type,
            "pIndex": page_index,
            "pSize": self.page_size,
            **params
        }
        
        try:
            response = requests.get(
                self.BASE_URL,
                params=request_params,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            
            if self.response_type == "json":
                return response.json()
            else:
                # XML 응답 처리 (필요시 추가)
                raise NotImplementedError("XML 응답 처리는 아직 구현되지 않았습니다.")
        
        except requests.exceptions.RequestException as e:
            print(f"❌ API 요청 실패: {e}")
            raise
    
    def search_meetings(
        self,
        dae_num: Optional[str] = None,
        conf_date: Optional[str] = None,
        comm_name: Optional[str] = None,
        class_name: Optional[str] = None,
        title: Optional[str] = None,
        sub_name: Optional[str] = None,
        dept_cd: Optional[str] = None,
        max_pages: Optional[int] = None
    ) -> List[CommitteeMeetingRecord]:
        """
        위원회 회의록 검색
        
        Args:
            dae_num: 대수 (필수)
            conf_date: 회의날짜 (필수)
            comm_name: 위원회명 (선택)
            class_name: 회의종류명 (선택)
            title: 회의명 (선택)
            sub_name: 안건명 (선택)
            dept_cd: 위원회코드 (선택)
            max_pages: 최대 페이지 수 (None이면 전체)
        
        Returns:
            회의록 레코드 리스트
        """
        if not dae_num or not conf_date:
            raise ValueError("DAE_NUM과 CONF_DATE는 필수 파라미터입니다.")
        
        params = {
            "DAE_NUM": dae_num,
            "CONF_DATE": conf_date
        }
        
        if comm_name:
            params["COMM_NAME"] = comm_name
        if class_name:
            params["CLASS_NAME"] = class_name
        if title:
            params["TITLE"] = title
        if sub_name:
            params["SUB_NAME"] = sub_name
        if dept_cd:
            params["DEPT_CD"] = dept_cd
        
        all_records = []
        page_index = 1
        
        while True:
            try:
                data = self._make_request(params, page_index)
                
                # 응답 구조 파싱 (실제 응답 구조에 맞게 수정 필요)
                records = self._parse_response(data)
                
                if not records:
                    break
                
                all_records.extend(records)
                
                # 최대 페이지 제한 확인
                if max_pages and page_index >= max_pages:
                    break
                
                # 다음 페이지가 있는지 확인
                if len(records) < self.page_size:
                    break
                
                page_index += 1
                time.sleep(self.request_delay)
            
            except Exception as e:
                print(f"  ❌ 페이지 {page_index} 처리 중 오류: {e}")
                break
        
        return all_records
    
    def _parse_response(self, data: Dict[str, Any]) -> List[CommitteeMeetingRecord]:
        """API 응답 파싱"""
        records = []
        
        # 국회 Open API 응답 구조:
        # { "ncwgseseafwbuheph": [{ "head": [...], "row": [{...}, {...}] }] }
        # 또는
        # { "ncwgseseafwbuheph": { "head": [...], "row": [{...}, {...}] } }
        
        if not isinstance(data, dict):
            return records
        
        # API 응답의 루트 키 찾기
        root_key = None
        for key in data.keys():
            if "ncwgseseafwbuheph" in key.lower():
                root_key = key
                break
        
        if not root_key:
            # 루트 키를 찾지 못한 경우, 직접 row 키 찾기
            if "row" in data:
                root_data = data
            else:
                return records
        else:
            root_data = data[root_key]
        
        # row 데이터 추출
        rows = []
        if isinstance(root_data, list):
            # 리스트인 경우: [{ "head": [...], "row": [...] }]
            for item in root_data:
                if isinstance(item, dict) and "row" in item:
                    if isinstance(item["row"], list):
                        rows.extend(item["row"])
                    else:
                        rows.append(item["row"])
        elif isinstance(root_data, dict):
            # 딕셔너리인 경우: { "head": [...], "row": [...] }
            if "row" in root_data:
                if isinstance(root_data["row"], list):
                    rows = root_data["row"]
                else:
                    rows = [root_data["row"]]
        
        # 레코드 생성
        for row in rows:
            if isinstance(row, dict):
                try:
                    records.append(CommitteeMeetingRecord.from_dict(row))
                except Exception as e:
                    print(f"  ⚠️  레코드 파싱 실패: {e}")
                    continue
        
        return records
    
    def get_all_meetings_by_session(
        self,
        session_name: str,
        year: Optional[int] = None
    ) -> List[CommitteeMeetingRecord]:
        """
        특정 회차의 모든 위원회 회의록 수집
        
        Args:
            session_name: 회차명 (예: "제415회")
            year: 연도 (None이면 session_name에서 추출)
        
        Returns:
            회의록 레코드 리스트
        """
        # 회차명에서 대수 추출 (예: "제415회" -> "22")
        # 실제 대수 매핑이 필요할 수 있음
        import re
        match = re.search(r"제\s*(\d+)\s*회", session_name)
        if not match:
            raise ValueError(f"회차명 형식이 올바르지 않습니다: {session_name}")
        
        session_num = int(match.group(1))
        # 대수 계산 (예: 415회 -> 22대, 실제 매핑 필요)
        # 임시로 22대 사용 (실제 매핑 테이블 필요)
        dae_num = "22"  # TODO: 실제 대수 매핑 로직 구현
        
        # 연도 추출
        if year is None:
            # 회차명이나 다른 정보에서 연도 추출 필요
            # 임시로 2024 사용
            year = 2024  # TODO: 실제 연도 추출 로직 구현
        
        conf_date = str(year)
        
        print(f"🔍 회차 {session_name} (대수: {dae_num}, 연도: {year}) 검색 중...")
        
        return self.search_meetings(
            dae_num=dae_num,
            conf_date=conf_date
        )

