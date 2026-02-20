"""
메타DB 관리 서비스
SQLite/PostgreSQL 기반 문서 메타데이터 관리
"""
import os
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.models.document import DocumentMetadata, DocumentResponse

Base = declarative_base()


class DocumentTable(Base):
    """문서 메타데이터 테이블"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    assembly_number = Column(String, nullable=False)
    session_type = Column(String, nullable=False)
    committee = Column(String, nullable=False)
    meeting_number = Column(Integer, nullable=False)
    date = Column(String, nullable=False)
    vector_store_file_id = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MetadataDBService:
    """메타DB 관리 서비스"""
    
    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "sqlite:///./metadata.db")
        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        self.engine = create_engine(db_url, connect_args=connect_args)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 초기화"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """세션 생성"""
        return self.SessionLocal()
    
    def create_document(self, metadata: DocumentMetadata) -> DocumentMetadata:
        """문서 메타데이터 생성"""
        db = self.get_session()
        try:
            db_doc = DocumentTable(
                filename=metadata.filename,
                assembly_number=metadata.assembly_number,
                session_type=metadata.session_type,
                committee=metadata.committee,
                meeting_number=metadata.meeting_number,
                date=metadata.date,
                vector_store_file_id=metadata.vector_store_file_id,
                is_active=metadata.is_active
            )
            db.add(db_doc)
            db.commit()
            db.refresh(db_doc)
            
            return DocumentMetadata(
                id=db_doc.id,
                filename=db_doc.filename,
                assembly_number=db_doc.assembly_number,
                session_type=db_doc.session_type,
                committee=db_doc.committee,
                meeting_number=db_doc.meeting_number,
                date=db_doc.date,
                vector_store_file_id=db_doc.vector_store_file_id,
                is_active=db_doc.is_active,
                created_at=db_doc.created_at,
                updated_at=db_doc.updated_at
            )
        finally:
            db.close()
    
    def get_document(self, doc_id: int) -> Optional[DocumentMetadata]:
        """문서 메타데이터 조회"""
        db = self.get_session()
        try:
            db_doc = db.query(DocumentTable).filter(DocumentTable.id == doc_id).first()
            if db_doc is None:
                return None
            
            return DocumentMetadata(
                id=db_doc.id,
                filename=db_doc.filename,
                assembly_number=db_doc.assembly_number,
                session_type=db_doc.session_type,
                committee=db_doc.committee,
                meeting_number=db_doc.meeting_number,
                date=db_doc.date,
                vector_store_file_id=db_doc.vector_store_file_id,
                is_active=db_doc.is_active,
                created_at=db_doc.created_at,
                updated_at=db_doc.updated_at
            )
        finally:
            db.close()
    
    def get_all_documents(
        self, 
        is_active: Optional[bool] = None,
        committee: Optional[str] = None,
        assembly_number: Optional[str] = None
    ) -> List[DocumentMetadata]:
        """문서 목록 조회"""
        db = self.get_session()
        try:
            query = db.query(DocumentTable)
            
            if is_active is not None:
                query = query.filter(DocumentTable.is_active == is_active)
            if committee:
                query = query.filter(DocumentTable.committee == committee)
            if assembly_number:
                query = query.filter(DocumentTable.assembly_number == assembly_number)
            
            db_docs = query.order_by(DocumentTable.date.desc()).all()
            
            return [
                DocumentMetadata(
                    id=doc.id,
                    filename=doc.filename,
                    assembly_number=doc.assembly_number,
                    session_type=doc.session_type,
                    committee=doc.committee,
                    meeting_number=doc.meeting_number,
                    date=doc.date,
                    vector_store_file_id=doc.vector_store_file_id,
                    is_active=doc.is_active,
                    created_at=doc.created_at,
                    updated_at=doc.updated_at
                )
                for doc in db_docs
            ]
        finally:
            db.close()
    
    def update_document(self, doc_id: int, metadata: DocumentMetadata) -> Optional[DocumentMetadata]:
        """문서 메타데이터 업데이트"""
        db = self.get_session()
        try:
            db_doc = db.query(DocumentTable).filter(DocumentTable.id == doc_id).first()
            if db_doc is None:
                return None
            
            db_doc.vector_store_file_id = metadata.vector_store_file_id
            db_doc.is_active = metadata.is_active
            db_doc.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(db_doc)
            
            return DocumentMetadata(
                id=db_doc.id,
                filename=db_doc.filename,
                assembly_number=db_doc.assembly_number,
                session_type=db_doc.session_type,
                committee=db_doc.committee,
                meeting_number=db_doc.meeting_number,
                date=db_doc.date,
                vector_store_file_id=db_doc.vector_store_file_id,
                is_active=db_doc.is_active,
                created_at=db_doc.created_at,
                updated_at=db_doc.updated_at
            )
        finally:
            db.close()
    
    def delete_document(self, doc_id: int) -> bool:
        """문서 메타데이터 삭제"""
        db = self.get_session()
        try:
            db_doc = db.query(DocumentTable).filter(DocumentTable.id == doc_id).first()
            if db_doc is None:
                return False
            
            db.delete(db_doc)
            db.commit()
            return True
        finally:
            db.close()
    
    def get_active_file_ids(self) -> List[str]:
        """Active Window의 Vector Store 파일 ID 목록 조회"""
        db = self.get_session()
        try:
            active_docs = db.query(DocumentTable).filter(
                DocumentTable.is_active == True,
                DocumentTable.vector_store_file_id.isnot(None)
            ).all()
            
            return [doc.vector_store_file_id for doc in active_docs if doc.vector_store_file_id]
        finally:
            db.close()
    
    def get_all_file_ids(self) -> List[str]:
        """모든 파일의 Vector Store 파일 ID 목록 조회 (Active + Inactive)"""
        db = self.get_session()
        try:
            all_docs = db.query(DocumentTable).filter(
                DocumentTable.vector_store_file_id.isnot(None)
            ).all()
            
            return [doc.vector_store_file_id for doc in all_docs if doc.vector_store_file_id]
        finally:
            db.close()




