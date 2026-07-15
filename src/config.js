import "dotenv/config";

// Ollama's OpenAI-compatible endpoint. Every LLM call in this app (chat
// and embeddings) is routed through this URL — no external API calls.
export const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://localhost:11434/v1";

// Ollama ignores the API key, but the OpenAI client requires a non-empty string.
export const OLLAMA_API_KEY = "ollama";

export const CHAT_MODEL = process.env.OLLAMA_CHAT_MODEL || "llama3.2";
export const EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL || "nomic-embed-text";

// Chunking parameters for the PDF ingestion tool.
export const CHUNK_SIZE = Number(process.env.CHUNK_SIZE || 1500);
export const CHUNK_OVERLAP = Number(process.env.CHUNK_OVERLAP || 200);

// Number of consecutive chunks grouped into one "section" for the map step,
// to reduce total LLM calls while keeping each call's context manageable.
export const SECTION_SIZE = Number(process.env.SECTION_SIZE || 5);

// Default top-k for vector similarity search.
export const RETRIEVER_K = Number(process.env.RETRIEVER_K || 6);
