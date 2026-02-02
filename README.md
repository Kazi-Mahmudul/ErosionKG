# ErosionKG - GraphRAG Knowledge Assistant

A premium full-stack GraphRAG application for soil erosion research, combining Next.js, FastAPI, Neo4j, and Groq LLM.

## Live: [ErosionKG](https://erosionkg.vercel.app)

![ErosionKG Dashboard](https://img.shields.io/badge/Status-Production%20Ready-green)
![License](https://img.shields.io/badge/License-MIT-blue)

## 🚀 Features

### Core Capabilities
- **GraphRAG Retrieval**: Hybrid search combining vector similarity + knowledge graph traversal
- **Real-time Streaming**: Server-Sent Events (SSE) for instant AI responses
- **Interactive Knowledge Graph**: Dynamic sub-graphs with node exploration and edge tooltips
- **Resizable Panels**: Drag-to-resize interface for optimal screen usage
- **Premium UI/UX**: Glassmorphism, gradient animations, and micro-interactions

### Advanced Features
- ✅ Numbered citations with click-to-open papers
- ✅ Copy & export responses as Markdown
- ✅ Metric dashboards with inline cards
- ✅ Node click → auto-fill chat queries
- ✅ Edge hover → view source papers
- ✅ Dark/light mode with smooth transitions

## 📁 Project Structure

```
erosion-kg/
├── app/                    # Next.js 15 frontend
│   ├── page.tsx           # Main 3-panel dashboard
│   ├── layout.tsx         # App shell with theme provider
│   └── globals.css        # Tailwind styles + custom theming
├── components/            # React components
│   ├── chat-interface.tsx
│   ├── graph-visualizer.tsx
│   ├── research-library.tsx
│   └── theme-*.tsx
├── api/                   # FastAPI backend
│   ├── index.py          # Main API routes
│   ├── kg/               # Knowledge graph modules
│   │   ├── graphrag_retriever.py
│   │   ├── entity_linker.py
│   │   └── cleanup_entities.py
│   └── requirements.txt
├── public/               # Static assets
├── .env                  # Environment variables
└── vercel.json          # Deployment config
```

## 🛠️ Tech Stack

**Frontend:**
- Next.js 15 (React 19)
- TypeScript
- Tailwind CSS 3
- react-force-graph-2d (graph visualization)
- lucide-react (icons)

**Backend:**
- FastAPI (Python 3.11+)
- Neo4j (knowledge graph database)
- Groq API (Llama 3.3 70B)
- Sentence Transformers (embeddings)
- RapidFuzz (fuzzy entity matching)

## 📦 Installation

### Prerequisites
- Node.js 18+
- Python 3.11+
- Neo4j Database (local or AuraDB)
- Groq API Key

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd erosion-kg
npm install
pip install -r api/requirements.txt
```

### 2. Environment Variables

Create `.env` in the root:

```env
# Neo4j Configuration
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# Groq API
GROQ_API_KEY=your-groq-api-key

# Optional: Hugging Face for embeddings
HF_TOKEN=your-hf-token
```

### 3. Run Development Servers

**Option A: Run both concurrently**
```bash
npm run dev:all
```

**Option B: Run separately**

Terminal 1 (Frontend):
```bash
npm run dev
```

Terminal 2 (Backend):
```bash
cd api
python index.py
```

Access at `http://localhost:3000`


## 📊 Knowledge Graph Setup

### Initial Data Load

1. Place PDFs in `api/data/pdf/`
2. Run extraction pipeline:

```bash
cd api
python kg/pdf_processor.py  # Extract text + metadata
python kg/entity_extractor.py  # Extract entities
python kg/upload_to_neo4j.py  # Upload to Neo4j
```


## 📖 API Endpoints

### `POST /api/chat`
**Real-time GraphRAG chat with SSE streaming**

Request:
```json
{
  "query": "What modulates rill erosion?"
}
```

Response (SSE):
```
data: {"type": "token", "content": "Erosion modulates..."}
data: {"type": "graph", "data": {"nodes": [...], "edges": [...]}}
data: {"type": "done"}
```

### `GET /api/metadata`
**Research library statistics**

Response:
```json
{
  "paper_count": 8,
  "entity_count": 2676,
  "relationship_count": 4293,
  "dois": [...]
}
```

## 🧪 Testing

```bash
# Frontend
npm run lint
npm run build

# Backend
cd api
pytest tests/
```

## 🐛 Troubleshooting

**Citations showing "Page: N/A"**
- Ensure PDFs have extractable page metadata
- Check `chunk.metadata.get("page_number")` in backend

**Graph not displaying**
- Verify Neo4j connection in `.env`
- Check browser console for errors
- Ensure entities are uploaded to Neo4j

**Slow responses**
- Reduce `top_k` in retrieval (default: 5)
- Use Groq's faster models (llama-3.1-70b)

## 📝 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- Research papers from Elsevier, MDPI, and SpringerNature
- Built with [Next.js](https://nextjs.org/), [FastAPI](https://fastapi.tiangolo.com/), [Neo4j](https://neo4j.com/)
- LLM powered by [Groq](https://groq.com/)

---

**Developed for advanced soil erosion research** 🌍
