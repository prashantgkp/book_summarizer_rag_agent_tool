import { retrieveRelevantChunks } from "../tools/vectorStoreTool.js";
import { logStep } from "../utils/logger.js";

// Curated queries used to pull thematically relevant passages from across
// the whole book, independent of chunk order. This is the "retrieval" half
// of RAG: instead of re-reading every chunk, the Synthesizer gets a
// focused, retrieved view of the book's themes to enrich the map-reduce
// section summaries.
const THEME_QUERIES = [
  "the main argument, thesis, or plot of the book",
  "the central characters, entities, or concepts discussed",
  "the most important themes, conclusions, or takeaways",
  "notable quotes, statistics, examples, or turning points",
];

/**
 * Retriever Agent: given a built vector store, fetches the passages most
 * relevant to a fixed set of thematic queries and returns them as a single
 * text block for the Synthesizer Agent to use.
 */
export async function runRetrieverAgent(store, { k = 4 } = {}) {
  logStep("Retriever", `Running ${THEME_QUERIES.length} thematic retrieval queries...`);
  const seen = new Set();
  const notes = [];

  for (const query of THEME_QUERIES) {
    const results = await retrieveRelevantChunks(store, query, k);
    for (const r of results) {
      const idx = r.metadata?.chunkIndex;
      if (idx !== undefined && seen.has(idx)) continue;
      if (idx !== undefined) seen.add(idx);
      notes.push(`- (chunk ${idx}) ${r.pageContent.slice(0, 400)}`);
    }
  }

  logStep("Retriever", `Retrieved ${notes.length} unique supporting passages.`);
  return notes.join("\n");
}

/** Ad-hoc retrieval helper, e.g. used by the Critic Agent to fact-check a claim. */
export async function retrieveForQuery(store, query, k = 4) {
  return retrieveRelevantChunks(store, query, k);
}
