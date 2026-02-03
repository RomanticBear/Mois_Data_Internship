"""
Active Window 관리 서비스
슬라이딩 윈도우 방식으로 최근 N회차 회의록만 유지
"""
import os
from typing import List
from app.services.metadata_db import MetadataDBService
from app.services.vector_store import VectorStoreService
from app.models.document import DocumentMetadata


class ActiveWindowService:
    """Active Window 관리"""
    
    def __init__(
        self,
        window_size: int = None,
        metadata_db: MetadataDBService = None,
        vector_store: VectorStoreService = None
    ):
        self.window_size = window_size or int(os.getenv("ACTIVE_WINDOW_SIZE", "5"))
        self.metadata_db = metadata_db or MetadataDBService()
        self.vector_store = vector_store or VectorStoreService()
    
    def add_document(self, metadata: DocumentMetadata) -> DocumentMetadata:
        """
        새 문서 추가 및 Active Window 관리
        
        Args:
            metadata: 추가할 문서 메타데이터
            
        Returns:
            추가된 문서 메타데이터
        """
        # 새 문서를 Active로 설정
        metadata.is_active = True
        
        # 메타DB에 저장
        created_metadata = self.metadata_db.create_document(metadata)
        
        # Active Window 크기 초과 시 가장 오래된 문서 비활성화
        self._maintain_window_size()
        
        return created_metadata
    
    def _maintain_window_size(self):
        """Active Window 크기 유지"""
        # Active 문서 목록 조회 (날짜순 정렬)
        active_docs = self.metadata_db.get_all_documents(is_active=True)
        
        # Active Window 크기 초과 시 가장 오래된 문서 비활성화
        if len(active_docs) > self.window_size:
            # 날짜순 정렬 (오래된 것부터)
            active_docs_sorted = sorted(active_docs, key=lambda x: x.date)
            
            # 초과분 비활성화
            deactivate_count = len(active_docs) - self.window_size
            for doc in active_docs_sorted[:deactivate_count]:
                self.deactivate_document(doc.id)
    
    def deactivate_document(self, doc_id: int) -> bool:
        """
        문서 비활성화 (Vector Store에서 제거, 메타DB에서는 유지)
        
        Args:
            doc_id: 비활성화할 문서 ID
            
        Returns:
            성공 여부
        """
        doc = self.metadata_db.get_document(doc_id)
        if doc is None:
            return False
        
        # Vector Store에서 파일 제거
        if doc.vector_store_file_id:
            try:
                self.vector_store.delete_file(doc.vector_store_file_id)
            except Exception as e:
                print(f"Vector Store에서 파일 삭제 실패: {e}")
        
        # 메타DB에서 비활성화 표시 (파일은 삭제하지 않음)
        doc.is_active = False
        doc.vector_store_file_id = None
        self.metadata_db.update_document(doc_id, doc)
        
        return True
    
    def reactivate_document(self, doc_id: int, file_path: str) -> bool:
        """
        비활성 문서 재활성화 (Vector Store에 다시 업로드)
        
        Args:
            doc_id: 재활성화할 문서 ID
            file_path: 원본 파일 경로
            
        Returns:
            성공 여부
        """
        doc = self.metadata_db.get_document(doc_id)
        if doc is None:
            return False
        
        # Vector Store에 다시 업로드
        try:
            file_id = self.vector_store.upload_file(file_path, doc)
            
            # 메타DB 업데이트
            doc.is_active = True
            doc.vector_store_file_id = file_id
            self.metadata_db.update_document(doc_id, doc)
            
            # Active Window 크기 유지
            self._maintain_window_size()
            
            return True
        except Exception as e:
            print(f"문서 재활성화 실패: {e}")
            return False
    
    def get_active_documents(self) -> List[DocumentMetadata]:
        """Active Window 문서 목록 조회"""
        return self.metadata_db.get_all_documents(is_active=True)
    
    def get_window_size(self) -> int:
        """현재 Active Window 크기 반환"""
        return self.window_size
    
    def set_window_size(self, size: int):
        """Active Window 크기 설정 및 자동 조정"""
        self.window_size = size
        self._maintain_window_size()





