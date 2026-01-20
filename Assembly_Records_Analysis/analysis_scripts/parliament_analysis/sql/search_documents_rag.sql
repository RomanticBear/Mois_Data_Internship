-- Supabase에서 실행할 SQL 함수
-- 벡터 유사도 검색을 서버 측에서 수행하여 성능 향상

CREATE OR REPLACE FUNCTION search_documents_rag(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 5,
    filter_session_name text DEFAULT NULL,
    filter_party_name text DEFAULT NULL,
    filter_agenda_title text DEFAULT NULL,
    filter_source_type text DEFAULT NULL
)
RETURNS TABLE (
    document_id uuid,
    content text,
    metadata jsonb,
    similarity float,
    source_type text,
    source_id text
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.document_id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) as similarity,
        d.source_type,
        d.source_id
    FROM documents_rag d
    WHERE 
        (1 - (d.embedding <=> query_embedding)) >= match_threshold
        AND (filter_session_name IS NULL OR d.metadata->>'session_name' = filter_session_name)
        AND (filter_party_name IS NULL OR d.metadata->>'party_name' = filter_party_name)
        AND (filter_agenda_title IS NULL OR d.metadata->>'agenda_title' = filter_agenda_title)
        AND (filter_source_type IS NULL OR d.source_type = filter_source_type)
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 인덱스 추가 (성능 향상)
CREATE INDEX IF NOT EXISTS idx_documents_rag_embedding ON documents_rag 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_documents_rag_source_type ON documents_rag (source_type);
CREATE INDEX IF NOT EXISTS idx_documents_rag_metadata_session ON documents_rag USING gin (metadata jsonb_path_ops);

COMMENT ON FUNCTION search_documents_rag IS '벡터 유사도 기반 문서 검색 함수. 서버 측에서 벡터 검색을 수행하여 성능을 향상시킵니다.';



