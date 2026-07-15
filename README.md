# Book Summarizer RAG

A multi-agent, multi-tool Retrieval-Augmented Generation (RAG) pipeline that
reads a PDF book and produces a structured Markdown (or JSON) summary,
running entirely against a **local Ollama LLM** via its OpenAI-compatible API.
No data leaves your machine.

## Architecture

```
                         ┌────────────────────┐
   book.pdf ───────────▶ │   Ingest (PDF Tool) │  load + chunk (overlapping)
                         └─────────┬──────────┘
                                   ▼
                         ┌────────────────────┐
                         │  Embed (Vector DB)  │  Ollama embeddings → MemoryVectorStore
                         └─────────┬──────────┘
                     ┌─────────────┴─────────────┐
                     ▼                            ▼
          ┌─────────────────────┐      ┌───────────────────────┐
          │  Summarizer Agent   │      │   Retriever Agent      │
          │  (map step, per     │      │   (thematic RAG        │
          │   section)          │      │    queries)            │
          └──────────┬──────────┘      └───────────┬───────────┘
                     └─────────────┬─────────────────┘
                                   ▼
                       ┌─────────────────────────┐
                       │ Synthesizer/Editor Agent │  reduce step → draft Markdown
                       └────────────┬─────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │     Critic / QA Agent    │  fact-checks draft against
                       └────────────┬─────────────┘  fresh retrieval, pass/revise
                             pass │   │ revise (bounded to 1 loop)
                                  │   └──────────────► back to Synthesizer
                                  ▼
                          final Markdown / JSON summary
```

The flow is implemented as an explicit **LangGraph** `StateGraph`
(`src/graph/orchestrator.js`), not a fully autonomous tool-calling agent.
This is a deliberate choice: local models via Ollama vary widely in how
reliably they support function-calling, so each "agent" is a graph node that
deterministically calls its underlying LangChain **tool(s)**. The same tools
(`pdf_extract`, `vector_search`, `summarize_chunk`, `synthesize_summary`) are
also exported as proper LangChain `Tool` objects (via `@langchain/core/tools`
`tool()`), so you can wire them into an autonomous `AgentExecutor` /
tool-calling agent instead, if your chosen Ollama model supports it well.

### Agents

| Agent | Role | File |
|---|---|---|
| **Retriever** | Runs curated thematic queries against the vector store to pull the most relevant passages from anywhere in the book (the actual "R" in RAG). | `src/agents/retrieverAgent.js` |
| **Summarizer** | Map step — summarizes the book section-by-section (groups of consecutive chunks), preserving concrete facts. | `src/agents/summarizerAgent.js` |
| **Synthesizer / Editor** | Reduce step — merges all section summaries + retrieved theme notes into one coherent, de-duplicated Markdown summary. | `src/agents/synthesizerAgent.js` |
| **Critic / QA** | Re-retrieves fresh evidence and checks the draft summary for accuracy/coherence/completeness; returns pass/revise, bounded to one revision loop. | `src/agents/criticAgent.js` |

### Tools

| Tool | Purpose | File |
|---|---|---|
| `pdf_extract` / `ingestPdf` | Parse PDF → raw text → overlapping chunks (`RecursiveCharacterTextSplitter`) | `src/tools/pdfTool.js` |
| `vector_search` / `buildVectorStore` | Embed chunks via Ollama, build an in-memory vector store, run similarity search | `src/tools/vectorStoreTool.js` |
| `summarize_chunk` | LLM call that summarizes one chunk/section | `src/tools/summarizeTool.js` |
| `synthesize_summary` | LLM call that reduces section summaries into the final Markdown | `src/tools/synthesizeTool.js` |

### Folder structure

```
book-summarizer-rag/
├── package.json
├── .env.example
├── summarize.js              # CLI entry point
└── src/
    ├── config.js              # env-driven config (models, chunk sizes, etc.)
    ├── llm.js                 # shared ChatOpenAI + OpenAIEmbeddings clients pointed at Ollama
    ├── tools/
    │   ├── pdfTool.js          # PDF Ingestion Tool
    │   ├── vectorStoreTool.js  # Embedding & Vector Store Tool
    │   ├── summarizeTool.js    # Chunk-summarization Tool
    │   └── synthesizeTool.js   # Final synthesis Tool
    ├── agents/
    │   ├── retrieverAgent.js
    │   ├── summarizerAgent.js
    │   ├── synthesizerAgent.js
    │   └── criticAgent.js
    ├── graph/
    │   └── orchestrator.js     # LangGraph StateGraph wiring it all together
    └── utils/
        └── logger.js
```

## Setup

### 1. Install and start Ollama

Install Ollama from https://ollama.com, then pull a chat model and an
embedding model:

```bash
ollama pull llama3.1            # or any other chat-capable model you prefer
ollama pull nomic-embed-text    # embedding model

ollama serve                    # starts the server on http://localhost:11434
```

Verify the OpenAI-compatible endpoint is up:

```bash
curl http://localhost:11434/v1/models
```

### 2. Install project dependencies

```bash
cd book-summarizer-rag
npm install
```

### 3. Configure environment (optional)

```bash
cp .env.example .env
# edit .env if you want a different model, chunk size, etc.
```

## Usage

```bash
# Print a Markdown summary to stdout
node summarize.js --file ./book.pdf

# Write JSON (includes chunk counts, critic verdict, revision count) to a file
node summarize.js --file ./book.pdf --format json --out ./summary.json

# Write Markdown to a file instead of stdout
node summarize.js --file ./book.pdf --out ./summary.md
```

Example progress log while running:

```
[+0.0s] Starting multi-agent RAG summarization pipeline for: /path/book.pdf
[+0.0s] Agents: Retriever -> Summarizer (map) -> Synthesizer (reduce) -> Critic (QA)
[+0.1s] Reading PDF: /path/book.pdf
[+0.4s] Parsed 214 pages, 412000 characters. Splitting into chunks...
[+0.6s] Created 331 chunks (chunkSize=1500, overlap=200).
[+0.6s] Embedding 331 chunks via Ollama (this may take a while)...
[+38.2s] Vector store ready.
[+38.2s] [Summarizer] Summarizing 67 section(s) from 331 chunk(s)...
[+38.2s] [Summarizer] Section 1/67...
...
[+310.5s] [Summarizer] All sections summarized.
[+310.5s] [Retriever] Running 4 thematic retrieval queries...
[+318.1s] [Retriever] Retrieved 14 unique supporting passages.
[+318.1s] [Synthesizer] Combining 67 section summaries into a draft...
[+341.7s] [Synthesizer] Draft summary complete.
[+341.7s] [Critic] Retrieving spot-check passages...
[+344.0s] [Critic] Evaluating draft summary...
[+349.2s] [Critic] Verdict: pass (0 issue(s))
[+349.2s] [Orchestrator] Finalizing summary.
[+349.2s] Pipeline complete in 349.2s.
```

## Output format

The default Markdown output always contains these sections:

```markdown
## Overview
## Key Themes
## Chapter-by-Chapter Breakdown
## Notable Details
```

`--format json` wraps that same Markdown summary alongside pipeline metadata:

```json
{
  "file": "...",
  "numPages": 214,
  "numChunks": 331,
  "numSections": 67,
  "critique": { "verdict": "pass", "issues": [] },
  "revisionCount": 0,
  "summary": "## Overview\n..."
}
```

## Configuration reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama's OpenAI-compatible endpoint |
| `OLLAMA_CHAT_MODEL` | `llama3.1` | Chat model used by all agents |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model used for the vector store |
| `CHUNK_SIZE` | `1500` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Character overlap between chunks |
| `SECTION_SIZE` | `5` | Chunks grouped per section for the map step |
| `RETRIEVER_K` | `6` | Default top-k for similarity search |

## Swapping in a persistent vector store

The pipeline uses LangChain's `MemoryVectorStore` by default so the project
has zero external service dependencies beyond Ollama itself. To use Chroma
or FAISS instead, install the relevant client and change only
`buildVectorStore()` in `src/tools/vectorStoreTool.js` — every other module
only depends on the shared `similaritySearch(query, k)` interface, so no
other file needs to change.

## Notes on the multi-agent design

- Larger books or slower models: increase `SECTION_SIZE` to reduce the
  number of LLM calls in the map step, at the cost of coarser section
  summaries.
- The Critic's revision loop is capped at `MAX_REVISIONS = 1`
  (`src/graph/orchestrator.js`) so a model that never returns a clean "pass"
  verdict can't loop forever.
- Every tool is also exported as a LangChain `Tool` object (`pdfExtractionTool`,
  `makeVectorSearchTool`, `chunkSummarizationTool`, `synthesisTool`) so you
  can plug them into `createToolCallingAgent` + `AgentExecutor` if you want a
  fully autonomous agent instead of the deterministic graph, for Ollama
  models with solid tool-calling support (e.g. `llama3.1`, `qwen2.5`).
