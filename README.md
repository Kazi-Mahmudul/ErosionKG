# ErosionKG

**ErosionKG** is an LLMOps-based knowledge graph system focused on landscape erosion. It is designed to ingest scientific literature, extract entities and relations defined by a research-grade ontology, and structure this information into a knowledge graph for analysis and evaluation.

## Project Structure

- **`ontology/`**: Contains the `erosion_ontology.yaml` defining entities (e.g., `ErosionProcess`, `Landform`, `Factor`) and relations.
- **`ingestion/`**: PDF extraction pipeline using **LlamaIndex** + **LlamaParse** (Multimodal Gemini 2.5 Flash).
    - `pipeline.py`: Main extraction script with structural parsing and contextual enrichment.
    - `schemas.py`: Pydantic models for structured output.
- **`data/`**: Directory for input PDFs and output JSON.
- **`api/`, `kg/`, `evaluation/`, `workflows/`**: Placeholder directories for future modules.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Kazi-Mahmudul/ErosionKG.git
    cd ErosionKG
    ```

2.  **Create a `.env` file**:
    Add your API keys:
    ```env
    GOOGLE_API_KEY=your_google_key
    LLAMA_CLOUD_API_KEY=your_llama_cloud_key
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

**Run the Extraction Pipeline**:
Place scientific PDFs in the `data/` folder and run:
```bash
python ingestion/pipeline.py
```
This will generate `data/extracted_chunks.json` containing high-fidelity markdown chunks, structured tables, and metadata.

## Ontology
The ontology is defined in `ontology/erosion_ontology.yaml` and includes research-grade definitions for landscape erosion domains.
