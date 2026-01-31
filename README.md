# ErosionKG

**ErosionKG** is an LLMOps-based knowledge graph system focused on landscape erosion. It is designed to ingest scientific literature, extract entities and relations defined by a research-grade ontology, and structure this information into a knowledge graph for analysis and evaluation. It features full metadata propagation (DOI, Page numbers, Citations) through the RAG pipeline.

## Project Structure

- **`ontology/`**: Contains the `erosion_ontology.yaml` defining entities (e.g., `ErosionProcess`, `Landform`, `Factor`) and relations.
- **`ingestion/`**: PDF extraction pipeline using **LlamaIndex** + **LlamaParse**.
    - `pipeline.py`: Main ingestion script extracting granular chunks with metadata (DOI, page numbers, citation strings).
    - `schemas.py`: Pydantic models for extracted data.
- **`extraction/`**: Knowledge Graph construction pipeline.
    - `kg_pipeline_manual.py`: Manual triplet extraction and Neo4j upsert pipeline.
- **`kg/`**: Knowledge Graph utilities.
    - `ingest_chunks.py`: Script to upsert processed chunks into Neo4j with full metadata.
    - `entity_resolution.py`: Semantic entity deduplication using fuzzy matching + LLM verification.
    - `vector_index.py`: Neo4j vector index management for chunks.
    - `graphrag_retriever.py`: Hybrid GraphRAG retriever with synonym expansion and multi-hop traversal.
    - `generation_engine.py`: Final RAG engine for answering queries with formatted citations and DOI links.
- **`prompts/`**: YAML-based prompt management system.
    - `registry.py`: Handles prompt versioning and loading.
    - `v1_baseline.yaml`: Baseline prompt configuration.
- **`data/`**: Directory for input PDFs and output JSON.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Kazi-Mahmudul/ErosionKG.git
    cd ErosionKG
    ```

2.  **Create a `.env` file**:
    ```env
    GOOGLE_API_KEY=your_google_key
    LLAMA_CLOUD_API_KEY=your_llama_cloud_key
    NEO4J_URI=neo4j+s://instance_id.databases.neo4j.io
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_password
    GROQ_API_KEY=your_groq_key
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Ingest PDFs & Metadata
Place scientific PDFs in `data/raw_papers/` and run:
```bash
python ingestion/pipeline.py
```
This generates `data/extracted_chunks.json` with DOI and page citations.

### 2. Ingest Chunks to Neo4j
Load the processed chunks into Neo4j:
```bash
python kg/ingest_chunks.py
```

### 3. Build Knowledge Graph
Extract entities and relations and ingest them into Neo4j:
```bash
python extraction/kg_pipeline_manual.py
```

### 4. Entity Resolution
Deduplicate entities using fuzzy matching + LLM verification:
```bash
python kg/entity_resolution.py --dry-run --threshold 85
python kg/entity_resolution.py --from-log  # Execute confirmed merges
```

### 5. Vector Indexing
Set up vector index for the granular chunks:
```bash
python kg/vector_index.py --setup
```

### 6. Generation (RAG)
Run the final RAG engine to get answers with full citations (DOI + Page Numbers):
```bash
python kg/generation_engine.py --query "What modulates rill erosion?" --no-stream
```

## Ontology
The ontology is defined in `ontology/erosion_ontology.yaml` and includes research-grade definitions for landscape erosion domains.
