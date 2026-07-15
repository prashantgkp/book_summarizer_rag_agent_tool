import { synthesizeSummary } from "../tools/synthesizeTool.js";
import { logStep } from "../utils/logger.js";

/**
 * Synthesizer/Editor Agent: the "reduce" step. Combines all section
 * summaries plus retrieved theme notes into one coherent Markdown summary.
 */
export async function runSynthesizerAgent(sectionSummaries, themeNotes) {
  logStep("Synthesizer", `Combining ${sectionSummaries.length} section summaries into a draft...`);
  const summary = await synthesizeSummary({ sectionSummaries, themeNotes });
  logStep("Synthesizer", "Draft summary complete.");
  return summary;
}

/** Called when the Critic Agent requests a revision, feeding its issues back in. */
export async function reviseSummary(sectionSummaries, themeNotes, critique) {
  logStep("Synthesizer", "Revising summary based on critic feedback...");
  const themeNotesWithCritique = `${themeNotes || ""}\n\nEditor must address this feedback:\n${critique}`;
  const summary = await synthesizeSummary({ sectionSummaries, themeNotes: themeNotesWithCritique });
  logStep("Synthesizer", "Revised summary complete.");
  return summary;
}
