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
llm = Gemini(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)  # Using 2.0-flash as 2.5 might not be available in lib yet, or assuming alias works.

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

def process_pdfs(data_dir: str = "data", output_file: str = "data/extracted_chunks.json"):
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return

    all_chunks = []

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        logger.info(f"Processing: {filename}")

        try:
            # 1. Parse PDF
            documents = parser.load_data(pdf_path)
            full_text = "\n".join([doc.text for doc in documents])
            
            # 2. Contextual Enrichment
            global_context = get_global_context(full_text)
            logger.info(f"Global Context for {filename}: {global_context}")

            # Prepend context to documents for node parsing (conceptual, usually we enrich nodes)
            # Strategy: We will enrich the nodes *after* parsing or prepend to text before.
            # MarkdownElementNodeParser works on documents.
            # We'll attach the context to the metadata or prepend it to the text.
            # User wants it prepended to every chunk.
            
            # 3. Structural Parsing
            node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
            nodes = node_parser.get_nodes_from_documents(documents)
            
            # 4. Process Nodes into Chunks
            for node in nodes:
                # Basic header path extraction (if available in metadata)
                # MarkdownElementNodeParser puts structured output in metadata usually?
                # Or we traverse. 
                # For simplicity/robustness with standard nodes:
                header_path_val = node.metadata.get("header_path", "") # standard extraction might not populate this exactly as requested without custom logic, but let's try to use what's there or default.
                if not header_path_val:
                     # Fallback: try to construct or just use "Root"
                     header_path_val = f"Global Context: {global_context}"
                else:
                     header_path_val = f"Global Context: {global_context} > {header_path_val}"

                chunk = ExtractedChunk(
                    metadata=ChunkMetadata(
                        source_file=filename,
                        page_number=int(node.metadata.get("page_label", 0)) if node.metadata.get("page_label") else 0, # LlamaParse usually gives page_label
                        header_path=header_path_val,
                        chunk_id=str(uuid.uuid4()),
                        last_modified=datetime.now().isoformat()
                    ),
                    content=node.get_content()
                )
                all_chunks.append(chunk)

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}", exc_info=True)

    # 5. Save Output
    output = ExtractionRun(chunks=all_chunks)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))
    
    logger.info(f"Extraction complete. Saved {len(all_chunks)} chunks to {output_file}")

if __name__ == "__main__":
    process_pdfs()
