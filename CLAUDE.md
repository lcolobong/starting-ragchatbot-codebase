# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Course Materials RAG (Retrieval-Augmented Generation) system. Full-stack web app that lets users query course materials via semantic search (ChromaDB + SentenceTransformers) and get AI-powered responses from Anthropic's Claude.

## Commands

**Install dependencies:**
```bash
uv sync
```

**Run the application:**
```bash
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Access points:**
- Web UI: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs

**Environment setup:** Create `.env` in project root with `ANTHROPIC_API_KEY=<key>`. See `.env.example`.

No test framework or linter is currently configured.

## Architecture

```
Frontend (vanilla HTML/CSS/JS)  →  FastAPI (app.py)  →  RAGSystem  →  Claude API
                                                          ├→ DocumentProcessor (parse/chunk docs)
                                                          ├→ VectorStore (ChromaDB, 2 collections)
                                                          ├→ SessionManager (in-memory history)
                                                          └→ AIGenerator + ToolManager (Claude tool_use)
```

**Query flow:** User submits question → FastAPI `/api/query` → RAGSystem → AIGenerator sends to Claude with tool definitions → Claude calls `CourseSearchTool` → VectorStore semantic search → Claude synthesizes answer from results → response with sources returned to frontend.

**Two ChromaDB collections:**
- `course_catalog`: course-level metadata for semantic course name resolution (e.g., "MCP" → "MCP: Build Rich-Context AI Apps")
- `course_content`: chunked course text for semantic search with lesson/course metadata filters

**Document format:** Course materials in `docs/` follow a structured text format with `Course Title:`, `Course Link:`, `Course Instructor:`, then `Lesson N: Title` sections. See existing files for the expected format.

**Tool architecture:** Abstract `Tool` base class with `get_tool_definition()` and `execute()` methods. `CourseSearchTool` is registered in `ToolManager`. Claude uses Anthropic's tool_use feature to invoke search during response generation.

## Key Configuration (backend/config.py)

- Model: `claude-sonnet-4-20250514`
- Embeddings: `all-MiniLM-L6-v2`
- Chunk size: 800 chars, 100 char overlap
- Max search results: 5
- Conversation history: 2 exchanges (4 messages)
- ChromaDB path: `./chroma_db` (relative to backend/)

## Tech Stack

- **Backend:** Python 3.13, FastAPI, ChromaDB, SentenceTransformers, Anthropic SDK
- **Frontend:** Vanilla JS, marked.js (CDN) for markdown rendering
- **Package manager:** uv (pyproject.toml + uv.lock)
