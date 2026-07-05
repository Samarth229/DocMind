from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer.")
    k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    source: str
    page: int | None
    rerank_score: float | None = None
    rrf_score: float | None = None


class CitationCheck(BaseModel):
    total_citations: int
    valid_citations: int
    invalid_citations: list[dict]
    all_valid: bool


class QueryResponse(BaseModel):
    answer: str
    chunks_used: list[ChunkResult]
    citation_check: CitationCheck


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int
    status: str


class DocumentsResponse(BaseModel):
    documents: list[str]
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    vectorstore_chunks: int
    ollama_model: str
