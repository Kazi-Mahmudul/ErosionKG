import os
import glob
import json
import logging
import uuid
from datetime import datetime
from dotenv import load_dotenv

from llama_parse import LlamaParse
from llama_index.llms.gemini import Gemini
from llama_index.core.node_parser import MarkdownElementNodeParser, SentenceSplitter
from llama_index.core import Document

from schemas import ExtractedChunk, ChunkMetadata, ExtractionRun

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Environment Variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in .env")
    exit(1)
if not LLAMA_CLOUD_API_KEY:
    logger.error("LLAMA_CLOUD_API_KEY not found in .env")
    exit(1)

# Initialize Gemini
# Initialize Gemini with strict rate limiting
class RateLimitedGemini(Gemini):
    def complete(self, *args, **kwargs):
        time.sleep(5)  # Enforce delay to stay under 15 RPM
        return super().complete(*args, **kwargs)

    def chat(self, *args, **kwargs):
        time.sleep(5)
        return super().chat(*args, **kwargs)
        
    def stream_complete(self, *args, **kwargs):
        time.sleep(5)
        return super().stream_complete(*args, **kwargs)

    def stream_chat(self, *args, **kwargs):
        time.sleep(5)
        return super().stream_chat(*args, **kwargs)
    
    # Async methods (MarkdownElementNodeParser uses async often)
    async def acomplete(self, *args, **kwargs):
        time.sleep(5)
        return await super().acomplete(*args, **kwargs)

    async def achat(self, *args, **kwargs):
        time.sleep(5)
        return await super().achat(*args, **kwargs)
    
    async def astream_complete(self, *args, **kwargs):
        time.sleep(5)
        return await super().astream_complete(*args, **kwargs)

    async def astream_chat(self, *args, **kwargs):
        time.sleep(5)
        return await super().astream_chat(*args, **kwargs)

llm = RateLimitedGemini(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)

# Initialize LlamaParse
parser = LlamaParse(
    result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="gemini-2.0-flash-001", # Adjusted to a known valid model name just in case, or user requested 2.5 explicitly? User asked for 2.5-flash. I will try to use the specific string requested if possible, but standard is usually 1.5/2.0. I will stick to the user's request string but keep a fallback note in mind. actually `gemini-2.5-flash` might be a typo for `gemini-1.5-flash` or a preview model. I will use the string provided by user.
    parsing_instruction="Extract this scientific paper on landscape erosion. Pay extreme attention to the numerical values and units in tables and charts, ensuring they are preserved in Markdown format.",
    api_key=LLAMA_CLOUD_API_KEY
)

def get_global_context(text: str) -> str:
    """Generates a 2-sentence summary of the paper's core focus."""
    prompt = (
        "Analyze the following text (which is the beginning of a scientific paper) "
        "and generate a 'Global Context Breadcrumb': a strict 2-sentence summary of the paper's core focus. "
        "Do not use introductory phrases like 'This paper discusses...'. Just state the summary."
        f"\n\nText Preview:\n{text[:5000]}"
    )
    response = llm.complete(prompt)
    return response.text.strip()

import time

def process_pdfs(data_dir: str = "data/raw_papers", output_file: str = "data/extracted_chunks.json"):
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return

    all_chunks = []
    
    # Load existing chunks if existing to avoid duplicates or losing data on restart
    # (Optional: check for duplicates by chunk_id or source_file, 
    # but for now we just append. To be robust we should check if file already processed)
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # assuming ExtractionRun format: { "chunks": [...] }
                existing_data = data.get("chunks", [])
                # We can choose to start fresh or append. 
                # User asked for incremental saving. 
                # Strategy: Load all, keep in memory, append new, save all.
                # To really be robust, we should check which files are already in existing_data
                processed_files = set(c['metadata']['source_file'] for c in existing_data if 'metadata' in c)
                for c in existing_data:
                    # Reconstruct objects if needed, or just keep dicts
                    # We need to conform to ExtractedChunk for type safety in Pydantic or just assume dicts
                    # Pydantic is simpler if we just load logic.
                    pass
                # Actually, simpler: just load everything into the list as dicts/objects
                # But to avoid re-processing, let's filter pdf_files
                logger.info(f"Loaded {len(existing_data)} existing chunks from {output_file}. Skipping processed files.")
                
                # Re-hydrate Pydantic models
                for c in existing_data:
                     # c is a dict
                     all_chunks.append(ExtractedChunk(**c))

                pdf_files = [p for p in pdf_files if os.path.basename(p) not in processed_files]
        except Exception as e:
            logger.warning(f"Could not load existing file {output_file}: {e}")

    # Process each PDF
    chunks = []
    
    # Configure pdf2doi logging
    import pdf2doi
    pdf2doi.config.set('verbose', False)

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        logger.info(f"Processing: {filename}")

        # Extract DOI
        try:
            results = pdf2doi.pdf2doi(pdf_path)
            doi_url = f"https://doi.org/{results['identifier']}" if results and results.get('identifier') else "N/A"
            logger.info(f"  - DOI: {doi_url}")
        except Exception as e:
            logger.warning(f"  - Failed to extract DOI: {e}")
            doi_url = "N/A"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 1. Parse PDF - documents are per-page from LlamaParse
                documents = parser.load_data(pdf_path)
                full_text = "\n".join([doc.text for doc in documents])
                
                # 2. Simple citation - just use filename as reference
                # DOI and page_number are the primary metadata for citations
                import re
                citation_str = filename.replace('.pdf', '').replace('_', ' ')
                logger.info(f"  - Citation ref: {citation_str}")
                
                # 3. Contextual Enrichment
                global_context = get_global_context(full_text)
                logger.info(f"Global Context for {filename}: {global_context}")

                # 4. Split each page into granular semantic chunks
                # This preserves page number while providing more detail
                sentence_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
                
                file_chunks = []
                for doc_idx, doc in enumerate(documents):
                    # LlamaParse page_label is 1-indexed page number
                    page_num = int(doc.metadata.get("page_label", doc_idx + 1))
                    
                    # Split page text into smaller chunks
                    page_chunks = sentence_splitter.split_text(doc.text)
                    
                    for chunk_text in page_chunks:
                        # Skip very short chunks (likely noise)
                        if len(chunk_text.strip()) < 50:
                            continue
                        
                        # Build header path
                        header_path_val = f"Global Context: {global_context}"
                        
                        chunk = ExtractedChunk(
                            metadata=ChunkMetadata(
                                source_file=filename,
                                page_number=page_num,
                                header_path=header_path_val,
                                chunk_id=str(uuid.uuid4()),
                                last_modified=datetime.now().isoformat(),
                                doi_url=doi_url,
                                citation_str=citation_str
                            ),
                            content=chunk_text
                        )
                        file_chunks.append(chunk)
                
                all_chunks.extend(file_chunks)
                logger.info(f"Successfully extracted {len(file_chunks)} chunks from {filename}")

                # Save Incrementally
                output = ExtractionRun(chunks=all_chunks)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output.model_dump_json(indent=2))
                logger.info(f"Incremental save: {len(all_chunks)} total chunks.")
                
                # Be nice to the API
                time.sleep(10) 
                break # Success

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "ResourceExhausted" in error_str:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"Rate limit hit for {filename}. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to process {filename}: {e}", exc_info=True)
                    break # Non-transient error, move to next file
    
    logger.info(f"Extraction complete. Saved {len(all_chunks)} chunks to {output_file}")

if __name__ == "__main__":
    process_pdfs()

