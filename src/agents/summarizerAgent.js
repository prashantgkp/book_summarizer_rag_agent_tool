import { summarizeText } from "../tools/summarizeTool.js";
import { SECTION_SIZE } from "../config.js";
import { logStep } from "../utils/logger.js";

/**
 * Groups consecutive chunks into larger "sections" so the pipeline makes
 * fewer LLM calls than one-per-chunk, while keeping each call's context
 * small enough for local models to handle comfortably.
 */
function groupIntoSections(chunks, sectionSize = SECTION_SIZE) {
  const sections = [];
  for (let i = 0; i < chunks.length; i += sectionSize) {
    sections.push(chunks.slice(i, i + sectionSize));
  }
  return sections;
}

/**
 * Summarizer Agent: the "map" step of map-reduce. Summarizes each section
 * of the book in sequence, logging progress as it goes.
 */
export async function runSummarizerAgent(chunks) {
  const sections = groupIntoSections(chunks);
  logStep("Summarizer", `Summarizing ${sections.length} section(s) from ${chunks.length} chunk(s)...`);

  const sectionSummaries = [];
  for (let i = 0; i < sections.length; i++) {
    const combinedText = sections[i].map((d) => d.pageContent).join("\n\n");
    logStep("Summarizer", `Section ${i + 1}/${sections.length}...`);
    const summary = await summarizeText(combinedText, { label: `${i + 1}` });
    sectionSummaries.push(summary);
  }

  logStep("Summarizer", "All sections summarized.");
  return sectionSummaries;
}
