# RAG-Based AI Teaching Assistant

A local **Retrieval-Augmented Generation (RAG)** pipeline that transforms video course content into an intelligent, searchable teaching assistant. Ask questions in natural language and receive accurate answers citing the **exact video number and timestamp**.

> Built with Whisper, Ollama (BGE-M3 + Llama 3.2), scikit-learn, FastAPI, and React.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Video → Transcript** | Extracts audio from videos and transcribes using OpenAI Whisper with translation support |
| 🔍 **Semantic Search** | Embeds transcript chunks via BGE-M3 and retrieves the most relevant segments using cosine similarity |
| 🤖 **AI Answers** | Generates human-like answers using Llama 3.2, citing exact video numbers and timestamps |
| 🌐 **Web Interface** | Modern React frontend with real-time backend health monitoring and chat UI |
| 🔌 **REST API** | FastAPI backend with `/ask`, `/health` endpoints and interactive Swagger docs |
| 🏠 **Fully Local** | Runs entirely on your machine — no cloud APIs, no data leaves your system |
| 🌍 **Multi-Language** | Configurable source language for Whisper (default: Hindi → English) |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────────┐
│   Videos    │────▶│  Audio (MP3) │────▶│  Transcripts │────▶│ Merged Chunks │
│  (videos/)  │     │  (audios/)   │     │   (jsons/)   │     │ (new_jsons/)  │
└─────────────┘     └──────────────┘     └──────────────┘     └───────┬───────┘
   video_to_mp3.py    mp3_to_json.py       merge_chunks.py           │
                                                          preprocess_new_json.py
                                                                     │
                                                                     ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────────┐
│  React UI   │◀───▶│  FastAPI     │────▶│  RAG Engine  │────▶│  Embeddings   │
│ (frontend/) │     │  (api.py)    │     │(rag_engine.py│     │  (.joblib)    │
└─────────────┘     └──────────────┘     └──────┬───────┘     └───────────────┘
                                                │
                                         ┌──────▼───────┐
                                         │   Ollama     │
                                         │ BGE-M3 embed │
                                         │ Llama 3.2    │
                                         └──────────────┘
```

---

## 📋 Prerequisites

1. **Python 3.10+** — [python.org](https://www.python.org/downloads/)
2. **ffmpeg** — required for audio extraction
   - **Windows:** `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/)
   - **macOS:** `brew install ffmpeg`
   - **Linux:** `sudo apt install ffmpeg`
3. **Ollama** — local LLM runtime: [ollama.com](https://ollama.com/)
   ```bash
   ollama pull bge-m3       # embedding model
   ollama pull llama3.2     # language model
   ```
4. **Node.js 18+** — for the frontend: [nodejs.org](https://nodejs.org/)

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/rag-ai-assistant.git
cd rag-ai-assistant

# Create virtual environment
python -m venv venv

# Activate
# Windows (PowerShell):
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
ollama serve
```

### 3. Run the Pipeline

Place your course videos in the `videos/` directory, then run each step:

```bash
python video_to_mp3.py          # Step 1: Extract audio
python mp3_to_json.py           # Step 2: Transcribe & translate
python merge_chunks.py          # Step 3: Merge segments (recommended)
python preprocess_new_json.py   # Step 4: Generate embeddings
```

### 4. Query via CLI

```bash
python process_incoming.py
# Ask a question: Where is this topic explained?
```

### 5. Start the Web Interface

```bash
# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000

# In a new terminal — start the frontend
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 📡 Pipeline Steps

| Step | Script | Description | Input → Output |
|------|--------|-------------|----------------|
| 1 | `video_to_mp3.py` | Extract audio tracks using ffmpeg | `videos/` → `audios/` |
| 2 | `mp3_to_json.py` | Transcribe & translate with Whisper | `audios/` → `jsons/` |
| 3 | `merge_chunks.py` | Merge small segments for better retrieval | `jsons/` → `new_jsons/` |
| 4a | `preprocess_json.py` | Embed raw segments (alternative) | `jsons/` → `embeddings.joblib` |
| 4b | `preprocess_new_json.py` | Embed merged chunks (recommended) | `new_jsons/` → `new_embeddings.joblib` |
| 5 | `process_incoming.py` | CLI Q&A interface | user query → answer |
| — | `api.py` | FastAPI server for web demo | HTTP API |

---

## 🔧 Configuration

All settings are configurable via environment variables:

### RAG Engine

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `EMBED_MODEL` | `bge-m3` | Embedding model name |
| `LLM_MODEL` | `llama3.2` | Language model for answer generation |
| `EMBEDDINGS_FILE` | `new_embeddings.joblib` | Path to embeddings index |
| `TOP_K_RESULTS` | `5` | Number of chunks to retrieve |

### Whisper Transcription

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large-v2` | Whisper model size |
| `WHISPER_LANGUAGE` | `hi` | Source audio language code |

### API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | CORS allowed origins (comma-separated) |

### Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `http://localhost:8000` | Backend API URL |

---

## 📡 API Reference

### `GET /`

Returns API info and links.

```json
{
  "message": "RAG Course Assistant API",
  "docs": "/docs",
  "health": "/health"
}
```

### `GET /health`

Returns backend health status.

```json
{
  "embeddings_loaded": true,
  "ollama_reachable": true,
  "chunk_count": 1234,
  "ready": true
}
```

### `POST /ask`

Submit a question and receive an AI-generated answer with source citations.

**Request:**
```json
{
  "question": "Where is this topic explained?"
}
```

**Response:**
```json
{
  "question": "Where is this topic explained?",
  "answer": "This topic is covered in Video 15 at timestamp 3:45...",
  "sources": [
    {
      "video_number": "15",
      "title": "Advanced Concepts",
      "start": "3:45",
      "end": "8:20",
      "score": 0.891,
      "excerpt": "In this section we discuss..."
    }
  ]
}
```

**Interactive docs:** Visit `http://localhost:8000/docs` for Swagger UI.

---

## 🧪 Testing

Run the full test suite:

```bash
pip install pytest httpx
pytest tests/ -v
```

The test suite covers:

| Test File | Coverage |
|-----------|----------|
| `tests/test_rag_engine.py` | Core RAG logic: embeddings, retrieval, prompt building, health checks |
| `tests/test_api.py` | FastAPI endpoints: `/`, `/health`, `/ask` with error scenarios |
| `tests/test_pipeline.py` | Pipeline utilities: filename parsing, chunk merging, batch processing |

---

## 🌐 Deployment

For deploying as a public demo (free):
- **Frontend** → Deploy `frontend/` to [Vercel](https://vercel.com)
- **Backend** → Run locally + expose via [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)

See **[DEPLOY.md](DEPLOY.md)** for step-by-step instructions.

---

## 📁 Project Structure

```
├── video_to_mp3.py          # Step 1: extract audio from videos
├── mp3_to_json.py           # Step 2: transcribe & translate with Whisper
├── merge_chunks.py          # Step 3: merge segments for retrieval
├── preprocess_json.py       # Step 4a: embed raw segments
├── preprocess_new_json.py   # Step 4b: embed merged chunks (recommended)
├── process_incoming.py      # Step 5: RAG Q&A (CLI)
├── rag_engine.py            # Core RAG logic (shared by CLI & API)
├── api.py                   # FastAPI server
├── requirements.txt         # Python dependencies
├── frontend/                # React + Vite + Tailwind frontend
│   ├── src/
│   │   ├── App.tsx          # Main chat application
│   │   ├── components/      # UI components (Header, ChatMessage, etc.)
│   │   └── lib/api.ts       # API client
│   ├── index.html           # HTML entry point
│   ├── package.json         # Node dependencies
│   └── vercel.json          # Vercel deployment config
├── tests/                   # pytest test suite
│   ├── test_rag_engine.py   # RAG engine unit tests
│   ├── test_api.py          # API endpoint tests
│   └── test_pipeline.py     # Pipeline utility tests
├── DEPLOY.md                # Public demo deployment guide
├── LICENSE                  # MIT License
├── videos/                  # Input videos (gitignored)
├── audios/                  # Extracted audio (gitignored)
├── jsons/                   # Whisper transcripts (gitignored)
└── new_jsons/               # Merged chunks (gitignored)
```

---

## 📝 Notes

- **Embeddings files** (`.joblib`) are gitignored — regenerate locally after cloning by running the pipeline.
- **`preprocess_json.py`** embeds raw Whisper segments into `embeddings.joblib` (fine-grained).
- **`preprocess_new_json.py`** embeds merged chunks into `new_embeddings.joblib` (recommended for Q&A).
- The pipeline is **course-agnostic** — use it with any video content in any Whisper-supported language.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
