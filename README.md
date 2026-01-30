# ErosionKG

**ErosionKG** is an LLMOps-based knowledge graph system focused on landscape erosion. It is designed to ingest scientific literature, extract entities and relations defined by a research-grade ontology, and structure this information into a knowledge graph for analysis and evaluation.

## Project Structure

- **`ontology/`**: Contains the `erosion_ontology.yaml` defining entities (e.g., `ErosionProcess`, `Landform`, `Factor`) and relations.
- **`ingestion/`**: PDF extraction pipeline using **LlamaIndex** + **LlamaParse** (Multimodal Gemini 2.0 Flash).
    - `pipeline.py`: Main ingestion script.
- **`extraction/`**: Knowledge Graph construction pipeline.
    - `kg_pipeline_manual.py`: Manual triplet extraction and Neo4j upsert pipeline.
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
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Ingest PDFs
Place scientific PDFs in `data/raw_papers/` and run:
```bash
python ingestion/pipeline.py
```
This processes PDFs into `data/extracted_chunks.json`.

### 2. Build Knowledge Graph
Extract entities and relations and ingest them into Neo4j:
```bash
python extraction/kg_pipeline_manual.py
```

### 3. Prompt Management
Manage prompt versions in `prompts/`. To switch versions, set the env var:
```bash
export PROMPT_VERSION=v1
```

## Ontology
The ontology is defined in `ontology/erosion_ontology.yaml` and includes research-grade definitions for landscape erosion domains.
