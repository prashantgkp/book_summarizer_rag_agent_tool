import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { ingestPdf } from "../tools/pdfTool.js";
import { buildVectorStore } from "../tools/vectorStoreTool.js";
import { runSummarizerAgent } from "../agents/summarizerAgent.js";
import { runRetrieverAgent } from "../agents/retrieverAgent.js";
import { runSynthesizerAgent, reviseSummary } from "../agents/synthesizerAgent.js";
import { runCriticAgent } from "../agents/criticAgent.js";
import { logStep } from "../utils/logger.js";

// ---------------------------------------------------------------------------
// Graph state definition. Each key is one slice of shared state that flows
// between nodes; nodes return partial updates and LangGraph merges them in.
// ---------------------------------------------------------------------------
const BookSummaryState = Annotation.Root({
  filePath: Annotation(),
  chunks: Annotation({ default: () => [] }),
  numPages: Annotation({ default: () => 0 }),
  vectorStore: Annotation(),
  sectionSummaries: Annotation({ default: () => [] }),
  themeNotes: Annotation({ default: () => "" }),
  draftSummary: Annotation({ default: () => "" }),
  critique: Annotation({ default: () => null }),
  revisionCount: Annotation({ default: () => 0 }),
  finalSummary: Annotation({ default: () => "" }),
});

const MAX_REVISIONS = 1; // caps the critic <-> synthesizer feedback loop

// ---------------------------------------------------------------------------
// Nodes — each wraps one tool or agent call.
// ---------------------------------------------------------------------------

/** Node: PDF Ingestion Tool (load + chunk). */
async function ingestNode(state) {
  const { chunks, numPages } = await ingestPdf(state.filePath);
  return { chunks, numPages };
}

/** Node: Embedding & Vector Store tool. */
async function embedNode(state) {
  const vectorStore = await buildVectorStore(state.chunks);
  return { vectorStore };
}

/** Node: Summarizer Agent (map step). */
async function mapSummarizeNode(state) {
  const sectionSummaries = await runSummarizerAgent(state.chunks);
  return { sectionSummaries };
}

/** Node: Retriever Agent (thematic RAG lookups). */
async function retrieveThemesNode(state) {
  const themeNotes = await runRetrieverAgent(state.vectorStore);
  return { themeNotes };
}

/** Node: Synthesizer/Editor Agent (reduce step). */
async function synthesizeNode(state) {
  const draftSummary = await runSynthesizerAgent(state.sectionSummaries, state.themeNotes);
  return { draftSummary };
}

/** Node: Critic/QA Agent. */
async function critiqueNode(state) {
  const critique = await runCriticAgent(state.vectorStore, state.draftSummary);
  return { critique };
}

/** Node: Synthesizer revision pass, triggered by a "revise" verdict. */
async function reviseNode(state) {
  const issues = (state.critique?.issues || []).join("; ");
  const draftSummary = await reviseSummary(state.sectionSummaries, state.themeNotes, issues);
  return { draftSummary, revisionCount: state.revisionCount + 1 };
}

/** Node: lock in the final summary. */
async function finalizeNode(state) {
  logStep("Orchestrator", "Finalizing summary.");
  return { finalSummary: state.draftSummary };
}

/** Conditional routing after the Critic: revise (bounded) or finalize. */
function routeAfterCritique(state) {
  if (state.critique?.verdict === "revise" && state.revisionCount < MAX_REVISIONS) {
    return "revise";
  }
  return "finalize";
}

// ---------------------------------------------------------------------------
// Graph assembly.
//
// Flow:
//   ingest -> embed -> { mapSummarize, retrieveThemes } -> synthesize
//          -> critiqueStep -> (revise -> critiqueStep) | finalize -> END
//
// mapSummarize and retrieveThemes both run off the embedded vector store and
// both feed into synthesize; LangGraph waits for all incoming edges of a
// node before running it, so synthesize only runs once both are done.
// ---------------------------------------------------------------------------
export function buildGraph() {
  const graph = new StateGraph(BookSummaryState)
    .addNode("ingest", ingestNode)
    .addNode("embed", embedNode)
    .addNode("mapSummarize", mapSummarizeNode)
    .addNode("retrieveThemes", retrieveThemesNode)
    .addNode("synthesize", synthesizeNode)
    .addNode("critiqueStep", critiqueNode)
    .addNode("revise", reviseNode)
    .addNode("finalize", finalizeNode)
    .addEdge(START, "ingest")
    .addEdge("ingest", "embed")
    .addEdge("embed", "mapSummarize")
    .addEdge("embed", "retrieveThemes")
    .addEdge("mapSummarize", "synthesize")
    .addEdge("retrieveThemes", "synthesize")
    .addEdge("synthesize", "critiqueStep")
    .addConditionalEdges("critiqueStep", routeAfterCritique, {
      revise: "revise",
      finalize: "finalize",
    })
    .addEdge("revise", "critiqueStep")
    .addEdge("finalize", END);

  return graph.compile();
}

/**
 * Runs the full pipeline for a given PDF path and returns the final graph
 * state, including the finished Markdown summary.
 */
export async function runBookSummaryPipeline(filePath) {
  const app = buildGraph();
  const result = await app.invoke({ filePath });
  return result;
}
