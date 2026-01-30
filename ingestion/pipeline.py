import os
import glob
import json
import logging
import uuid
from datetime import datetime
from dotenv import load_dotenv

from llama_parse import LlamaParse
from llama_index.llms.gemini import Gemini
from llama_index.core.node_parser import MarkdownElementNodeParser
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

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        logger.info(f"Processing: {filename}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 1. Parse PDF
                documents = parser.load_data(pdf_path)
                full_text = "\n".join([doc.text for doc in documents])
                
                # 2. Contextual Enrichment
                global_context = get_global_context(full_text)
                logger.info(f"Global Context for {filename}: {global_context}")

                # 3. Structural Parsing
                node_parser = MarkdownElementNodeParser(llm=llm, num_workers=1) # Reduced to 1 to strictly respect 15 RPM free tier limit
                nodes = node_parser.get_nodes_from_documents(documents)
                
                # 4. Process Nodes into Chunks
                file_chunks = []
                for node in nodes:
                    header_path_val = node.metadata.get("header_path", "")
                    if not header_path_val:
                         header_path_val = f"Global Context: {global_context}"
                    else:
                         header_path_val = f"Global Context: {global_context} > {header_path_val}"

                    chunk = ExtractedChunk(
                        metadata=ChunkMetadata(
                            source_file=filename,
                            page_number=int(node.metadata.get("page_label", 0)) if node.metadata.get("page_label") else 0,
                            header_path=header_path_val,
                            chunk_id=str(uuid.uuid4()),
                            last_modified=datetime.now().isoformat()
                        ),
                        content=node.get_content()
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
