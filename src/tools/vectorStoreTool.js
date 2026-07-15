import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { embeddings } from "../llm.js";
import { RETRIEVER_K } from "../config.js";
import { log } from "../utils/logger.js";

/**
 * Builds an in-memory vector store from chunked Documents, embedding each
 * chunk via the local Ollama embedding model.
 *
 * This uses LangChain's MemoryVectorStore to keep the project dependency-free
 * (no external vector DB process required). To swap in Chroma or FAISS for a
 * persistent store, install `@langchain/community` + the relevant client
 * (e.g. `chromadb`) and replace just this function — everything downstream
 * only relies on the shared `similaritySearch(query, k)` interface.
 *
 * Example Chroma swap:
 *   import { Chroma } from "@langchain/community/vectorstores/chroma";
 *   return Chroma.fromDocuments(docs, embeddings, { collectionName: "book" });
 */
export async function buildVectorStore(docs) {
  log(`Embedding ${docs.length} chunks via Ollama (${docs.length > 200 ? "this may take a while" : "please wait"})...`);
  const store = await MemoryVectorStore.fromDocuments(docs, embeddings);
  log("Vector store ready.");
  return store;
}

/**
 * Plain async helper used internally by agents (Retriever, Critic) to run a
 * similarity search against the store.
 */
export async function retrieveRelevantChunks(store, query, k = RETRIEVER_K) {
  return store.similaritySearch(query, k);
}

/**
 * The same retrieval capability exposed as a LangChain Tool, bound to a
 * specific store instance. Useful if you want an autonomous agent to decide
 * for itself when and what to search for, rather than the fixed queries the
 * deterministic Retriever Agent uses by default.
 */
export function makeVectorSearchTool(store) {
  return tool(
    async ({ query, k }) => {
      const results = await retrieveRelevantChunks(store, query, k ?? RETRIEVER_K);
      return JSON.stringify(
        results.map((r) => ({ chunkIndex: r.metadata?.chunkIndex, content: r.pageContent }))
      );
    },
    {
      name: "vector_search",
      description: "Search the book's vector store for the chunks most relevant to a query.",
      schema: z.object({
        query: z.string().describe("Natural language query or topic to search for"),
        k: z.number().optional().describe("Number of chunks to return (default 6)"),
      }),
    }
  );
}
