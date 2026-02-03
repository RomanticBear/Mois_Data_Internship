"""
파일명 파싱 유틸리티
국회회의록 PDF 파일명에서 메타데이터 추출
"""
import re
from typing import Optional, Dict
from app.models.document import DocumentMetadata


def parse_filename(filename: str) -> Dict[str, Optional[str | int]]:
    """
    파일명에서 메타데이터 추출
    
    예시 파일명:
    - "제22대국회 제415회(임시회) 제1차 행정안전위원회(전체회의) (2024.06.13.) (2).PDF"
    - "제22대국회 제415회(임시회) 제2차 행정안전위원회(전체회의) (2024.06.19.).PDF"
    
    Returns:
        추출된 메타데이터 딕셔너리
    """
    result = {
        "assembly_number": None,
        "session_type": None,
        "committee": None,
        "meeting_number": None,
        "date": None
    }
    
    # 국회 회차 추출 (예: "제415회")
    assembly_match = re.search(r'제(\d+)회', filename)
    if assembly_match:
        result["assembly_number"] = f"제{assembly_match.group(1)}회"
    
    # 회기 유형 추출 (예: "임시회", "정기회")
    session_match = re.search(r'\(([^)]+)\)', filename)
    if session_match:
        session_text = session_match.group(1)
        if "임시회" in session_text or "정기회" in session_text:
            result["session_type"] = session_text.split(")")[0] if ")" in session_text else session_text
    
    # 회의 번호 추출 (예: "제1차", "제2차")
    meeting_match = re.search(r'제(\d+)차', filename)
    if meeting_match:
        result["meeting_number"] = int(meeting_match.group(1))
    
    # 날짜 추출 (예: "2024.06.13.")
    date_match = re.search(r'(\d{4}\.\d{2}\.\d{2})', filename)
    if date_match:
        result["date"] = date_match.group(1)
    
    # 위원회 추출 (예: "행정안전위원회")
    # 위원회 이름은 "위원회"로 끝나는 패턴 찾기
    committee_patterns = [
        r'([^()]+위원회)',
        r'([가-힣]+위원회)'
    ]
    for pattern in committee_patterns:
        committee_match = re.search(pattern, filename)
        if committee_match:
            committee_text = committee_match.group(1)
            # 괄호 안의 내용 제거
            committee_text = re.sub(r'\([^)]*\)', '', committee_text).strip()
            if "위원회" in committee_text:
                result["committee"] = committee_text
                break
    
    return result


def create_metadata_from_filename(filename: str) -> Dict:
    """
    파일명으로부터 DocumentMetadata 딕셔너리 생성
    
    Args:
        filename: 파일명
        
    Returns:
        DocumentMetadata 생성에 필요한 딕셔너리
    """
    parsed = parse_filename(filename)
    
    # 기본값 설정
    metadata = {
        "filename": filename,
        "assembly_number": parsed.get("assembly_number") or "제415회",
        "session_type": parsed.get("session_type") or "임시회",
        "committee": parsed.get("committee") or "행정안전위원회",
        "meeting_number": parsed.get("meeting_number") or 1,
        "date": parsed.get("date") or "",
        "vector_store_file_id": None,
        "is_active": True
    }
    
    return metadata





