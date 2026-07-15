import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { makeChatModel } from "../llm.js";

const SYNTHESIZER_SYSTEM_PROMPT = `You are an editor who combines a series of section
summaries from one book into a single coherent book summary. Requirements:
- Merge overlapping or redundant points across sections.
- Keep a consistent, neutral, third-person tone throughout.
- Preserve the chronological / logical order of the book's content.
- Do not invent facts that are not implied by the provided section summaries.
Return the result as Markdown with exactly these headings, in this order:
## Overview
## Key Themes
## Chapter-by-Chapter Breakdown
## Notable Details
Under "Chapter-by-Chapter Breakdown", use one bullet per section provided.`;

/**
 * Reduce step: combines section summaries plus optional retrieved theme
 * notes into one structured Markdown book summary.
 */
export async function synthesizeSummary({ sectionSummaries, themeNotes }) {
  const model = makeChatModel({ temperature: 0.3 });
  const sectionsBlock = sectionSummaries.map((s, i) => `Section ${i + 1}: ${s}`).join("\n\n");

  const messages = [
    new SystemMessage(SYNTHESIZER_SYSTEM_PROMPT),
    new HumanMessage(
      `Section summaries:\n${sectionsBlock}\n\n` +
        (themeNotes ? `Additional retrieved theme notes:\n${themeNotes}\n\n` : "") +
        `Write the final Markdown book summary now.`
    ),
  ];
  const response = await model.invoke(messages);
  return response.content.toString().trim();
}

export const synthesisTool = tool(
  async ({ sectionSummaries, themeNotes }) => synthesizeSummary({ sectionSummaries, themeNotes }),
  {
    name: "synthesize_summary",
    description: "Combine multiple section summaries (and optional theme notes) into one coherent Markdown book summary.",
    schema: z.object({
      sectionSummaries: z.array(z.string()),
      themeNotes: z.string().optional(),
    }),
  }
);
