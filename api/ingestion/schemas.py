from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ChunkMetadata(BaseModel):
    source_file: str = Field(..., description="The name of the source PDF file.")
    page_number: int = Field(..., description="The page number where the chunk is located.")
    header_path: str = Field(..., description="The breadcrumb path of headers leading to this chunk.")
    chunk_id: str = Field(..., description="Unique identifier for the chunk.")
    last_modified: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of extraction.")
    doi_url: Optional[str] = Field(None, description="DOI URL of the source paper.")
    citation_str: Optional[str] = Field(None, description="Author and Year citation string.")

class ExtractedChunk(BaseModel):
    metadata: ChunkMetadata
    content: str = Field(..., description="The text content of the chunk.")

class ExtractionRun(BaseModel):
    chunks: List[ExtractedChunk]
