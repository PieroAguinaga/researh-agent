create extension if not exists vector;          -- pgvector: semantic search
create extension if not exists "uuid-ossp";     -- UUID generation helpers


-- -----------------------------------------------------------------------------
-- 1. conversation_history
--    Persistent memory for LangGraph's checkpointer.
--    Each row is one message turn within a thread (session).
-- -----------------------------------------------------------------------------
create table if not exists conversation_history (
    id              uuid        primary key default uuid_generate_v4(),
    thread_id       text        not null,           -- maps to LangGraph thread_id / session_id
    role            text        not null            -- 'human' | 'ai' | 'tool'
                                check (role in ('human', 'ai', 'tool')),
    content         text        not null,
    tool_calls      jsonb       default '[]'::jsonb, -- tool invocations made in this turn
    created_at      timestamptz default now()
);

-- Index for fast per-thread retrieval 
create index if not exists idx_conversation_thread
    on conversation_history (thread_id, created_at asc);


-- -----------------------------------------------------------------------------
-- 2. papers
--    Normalised metadata for every paper retrieved by the search agent.
-- -----------------------------------------------------------------------------
create table if not exists papers (
    id              uuid        primary key default uuid_generate_v4(),
    paper_id        text        not null unique,    -- arXiv ID or Semantic Scholar paperId
    title           text        not null,
    authors         text[]      default '{}',
    abstract        text        default '',
    url             text        default '',
    pdf_url         text        default '',
    published       date        default '',         -- ISO date string "2024-03-15"
    source          text        not null            -- 'arxiv' | 'semantic_scholar'
                                check (source in ('arxiv', 'semantic_scholar')),
    categories      text[]      default '{}',
    citation_count  integer     default 0,
    created_at      timestamptz default now()
);

create index if not exists idx_papers_source   on papers (source);
create index if not exists idx_papers_published on papers (published desc);


-- -----------------------------------------------------------------------------
-- 3. paper_embeddings
--    Vector store for RAG — one row per paper, embedding of title + abstract.
--    Dimension 1536 matches text-embedding-ada-002.
-- -----------------------------------------------------------------------------
create table if not exists paper_embeddings (
    id          uuid    primary key default uuid_generate_v4(),
    paper_id    text    not null references papers(paper_id) on delete cascade,
    content     text    not null,                   -- text that was embedded
    metadata    jsonb   default '{}'::jsonb,
    embedding   vector(1536)                        -- text-embedding-ada-002 output
);

-- IVFFlat index — fast approximate nearest-neighbour search (cosine distance)
-- Rebuild this index if the table grows beyond ~100k rows for best performance.
create index if not exists idx_paper_embeddings_vector
    on paper_embeddings
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);


-- -----------------------------------------------------------------------------
-- 4. match_paper_embeddings — RPC function used by LangChain SupabaseVectorStore
--    Called automatically by the RAG pipeline. Do not rename.
-- -----------------------------------------------------------------------------
create or replace function match_paper_embeddings (
    query_embedding  vector(1536),
    match_count      int     default 5,
    filter           jsonb   default '{}'::jsonb
)
returns table (
    id          uuid,
    paper_id    text,
    content     text,
    metadata    jsonb,
    similarity  float
)
language plpgsql
as $$
begin
    return query
    select
        pe.id,
        pe.paper_id,
        pe.content,
        pe.metadata,
        1 - (pe.embedding <=> query_embedding) as similarity
    from paper_embeddings pe
    where pe.metadata @> filter
    order by pe.embedding <=> query_embedding
    limit match_count;
end;
$$;
