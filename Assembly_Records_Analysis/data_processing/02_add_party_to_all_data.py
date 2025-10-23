#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국회 회의록 전체 데이터에 정당 정보 추가 스크립트
- 모든 회차의 발언 데이터에 정당 컬럼 추가
- 원본 데이터는 보존하고 새로운 폴더에 저장
"""

import pandas as pd
import os
import shutil
from collections import defaultdict

class PartyDataProcessor:
    def __init__(self):
        # 정당 정보 매핑
        self.party_mapping = {
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
        
        # 원본 데이터 경로
        self.source_path = "data/original"
        # 정당 정보가 추가된 데이터 저장 경로
        self.target_path = "data/with_party"
    
    def add_party_to_speeches(self, df):
        """발언 데이터에 정당 정보 추가"""
        df['party'] = df['speaker_name'].map(self.party_mapping).fillna('미분류')
        return df
    
    def add_party_to_header(self, df):
        """헤더 데이터는 정당 정보가 필요 없으므로 그대로 반환"""
        return df
    
    def process_session(self, session_folder):
        """특정 회차의 데이터 처리"""
        print(f"\n=== {session_folder} 처리 중 ===")
        
        source_session_path = os.path.join(self.source_path, session_folder)
        target_session_path = os.path.join(self.target_path, session_folder)
        
        # 대상 폴더 생성
        os.makedirs(target_session_path, exist_ok=True)
        
        # 해당 회차의 모든 파일 처리
        files = os.listdir(source_session_path)
        processed_files = 0
        
        for file in files:
            file_path = os.path.join(source_session_path, file)
            
            if file.endswith('_speeches.csv'):
                # 발언 데이터 처리
                print(f"  발언 데이터 처리: {file}")
                df = pd.read_csv(file_path)
                df_with_party = self.add_party_to_speeches(df)
                df_with_party.to_csv(os.path.join(target_session_path, file), index=False, encoding='utf-8-sig')
                processed_files += 1
                
            elif file.endswith('_header_summary.csv'):
                # 헤더 데이터 처리 (정당 정보 추가 없이 복사)
                print(f"  헤더 데이터 복사: {file}")
                df = pd.read_csv(file_path)
                df.to_csv(os.path.join(target_session_path, file), index=False, encoding='utf-8-sig')
                processed_files += 1
        
        print(f"  처리된 파일 수: {processed_files}개")
        return processed_files
    
    def analyze_all_speakers(self):
        """전체 발언자 분석"""
        print(f"\n=== 전체 발언자 분석 ===")
        
        all_speakers = set()
        party_stats = defaultdict(int)
        session_stats = {}
        
        # 모든 회차 폴더 처리
        sessions = [d for d in os.listdir(self.source_path) if d.startswith('제') and d.endswith('회')]
        sessions.sort()
        
        for session in sessions:
            source_session_path = os.path.join(self.source_path, session)
            session_speakers = set()
            session_party_stats = defaultdict(int)
            
            # 해당 회차의 모든 발언 파일 분석
            files = os.listdir(source_session_path)
            speech_files = [f for f in files if f.endswith('_speeches.csv')]
            
            for file in speech_files:
                file_path = os.path.join(source_session_path, file)
                df = pd.read_csv(file_path)
                
                # 발언자 수집
                speakers = df['speaker_name'].unique()
                all_speakers.update(speakers)
                session_speakers.update(speakers)
                
                # 정당별 통계
                for speaker in speakers:
                    party = self.party_mapping.get(speaker, '미분류')
                    party_stats[party] += 1
                    session_party_stats[party] += 1
            
            session_stats[session] = {
                'speakers': len(session_speakers),
                'party_stats': dict(session_party_stats)
            }
        
        print(f"전체 발언자 수: {len(all_speakers)}명")
        print("전체 발언자 목록:")
        for speaker in sorted(all_speakers):
            party = self.party_mapping.get(speaker, '미분류')
            print(f"  {speaker}: {party}")
        
        print(f"\n전체 정당별 발언자 수:")
        for party, count in party_stats.items():
            print(f"  {party}: {count}명")
        
        print(f"\n회차별 통계:")
        for session, stats in session_stats.items():
            print(f"  {session}: {stats['speakers']}명")
            for party, count in stats['party_stats'].items():
                print(f"    {party}: {count}명")
        
        return all_speakers, party_stats, session_stats
    
    def add_party_to_all_sessions(self):
        """모든 회차 데이터에 정당 정보 추가"""
        print("=== 모든 회차 데이터에 정당 정보 추가 시작 ===")
        
        # 대상 폴더 생성
        os.makedirs(self.target_path, exist_ok=True)
        
        # 모든 회차 폴더 처리
        sessions = [d for d in os.listdir(self.source_path) if d.startswith('제') and d.endswith('회')]
        sessions.sort()
        
        print(f"발견된 회차: {sessions}")
        
        total_processed = 0
        for session in sessions:
            processed = self.process_session(session)
            total_processed += processed
        
        print(f"\n=== 정당 정보 추가 완료 ===")
        print(f"총 처리된 파일 수: {total_processed}개")
        print(f"원본 데이터: {self.source_path}")
        print(f"정당 정보 추가된 데이터: {self.target_path}")
        
        return total_processed

def main():
    """메인 실행 함수"""
    print("국회 회의록 전체 데이터에 정당 정보 추가 시작")
    
    # 처리 객체 생성
    processor = PartyDataProcessor()
    
    # 전체 발언자 분석 (처리 전)
    print("처리 전 전체 발언자 분석")
    all_speakers, party_stats, session_stats = processor.analyze_all_speakers()
    
    # 모든 회차 데이터에 정당 정보 추가
    total_processed = processor.add_party_to_all_sessions()
    
    print(f"\n정당 정보 추가 완료!")
    print(f"총 처리된 파일 수: {total_processed}개")
    print(f"생성된 폴더: {processor.target_path}")

if __name__ == "__main__":
    main()
