import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { makeChatModel } from "../llm.js";

const SUMMARIZER_SYSTEM_PROMPT = `You are a meticulous book-summarization assistant.

Read the excerpt carefully and produce a summary that captures every key point: main events, arguments, or ideas; important names and characters; specific numbers, dates, and facts; and how they connect to one another. Preserve the accuracy of concrete details rather than generalizing them away.

Do not add opinions, interpretations, or information not present in the excerpt. Do not repeat these instructions in your output.

Write the summary as plain prose, covering the excerpt's content comprehensively while remaining concise.`;

/**
 * Summarizes a single chunk or a concatenated group of chunks ("section").
 * This is the map step of the map-reduce pipeline, called once per section
 * by the Summarizer Agent.
 */
export async function summarizeText(text, { label } = {}) {
  const model = makeChatModel({ temperature: 0.2 });
  const messages = [
    new SystemMessage(SUMMARIZER_SYSTEM_PROMPT),
    new HumanMessage(
      `${label ? `Section: ${label}\n\n` : ""}Excerpt:\n"""\n${text}\n"""\n\nSummary:`
    ),
  ];
  const response = await model.invoke(messages);
  return response.content.toString().trim();
}

/** LangChain Tool wrapper, for use in a tool-calling agent variant. */
export const chunkSummarizationTool = tool(
  async ({ text, label }) => summarizeText(text, { label }),
  {
    name: "summarize_chunk",
    description: "Produce a concise, faithful summary of a chunk (or group of chunks) of book text.",
    schema: z.object({
      text: z.string().describe("The chunk text to summarize"),
      label: z.string().optional().describe("Optional label, e.g. a section number"),
    }),
  }
);
