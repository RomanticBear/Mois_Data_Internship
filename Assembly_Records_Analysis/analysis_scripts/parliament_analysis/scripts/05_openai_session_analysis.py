#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 `analysis_scripts/05_openai_session_analysis.py` 엔트리포인트를
패키지 내부로 이동한 래퍼입니다. 실제 구현은
`parliament_analysis.run_session_analysis`에 존재합니다.
"""

from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
PACKAGE_ROOT = CURRENT_FILE.parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parliament_analysis.run_session_analysis import main  # noqa: E402


if __name__ == "__main__":
    main()

