import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { makeChatModel } from "../llm.js";
import { retrieveForQuery } from "./retrieverAgent.js";
import { logStep } from "../utils/logger.js";

const CRITIC_SYSTEM_PROMPT = `You are a strict QA reviewer checking a book summary
for accuracy, coherence, and completeness against the source excerpts provided.
Respond ONLY with a JSON object of the form:
{"verdict": "pass" | "revise", "issues": ["short issue 1", "short issue 2"]}
"issues" must be an empty array when verdict is "pass". Be concise. Do not include
any text outside the JSON object.`;

/**
 * Critic/QA Agent: spot-checks the draft summary against a handful of
 * passages retrieved fresh from the vector store (independent of the
 * section summaries the Synthesizer already saw), then returns a
 * pass/revise verdict with concrete issues for the Synthesizer to address.
 */
export async function runCriticAgent(store, draftSummary) {
  logStep("Critic", "Retrieving spot-check passages...");
  const spotChecks = await retrieveForQuery(store, "key facts and conclusions of the book", 5);
  const evidence = spotChecks.map((d) => `- ${d.pageContent.slice(0, 400)}`).join("\n");

  const model = makeChatModel({ temperature: 0 });
  const messages = [
    new SystemMessage(CRITIC_SYSTEM_PROMPT),
    new HumanMessage(
      `Draft summary:\n"""\n${draftSummary}\n"""\n\nSource excerpts for verification:\n${evidence}\n\nEvaluate the draft now.`
    ),
  ];

  logStep("Critic", "Evaluating draft summary...");
  const response = await model.invoke(messages);
  const raw = response.content.toString().trim();

  try {
    const cleaned = raw.replace(/```json|```/g, "").trim();
    const parsed = JSON.parse(cleaned);
    logStep("Critic", `Verdict: ${parsed.verdict} (${parsed.issues?.length ?? 0} issue(s))`);
    return parsed;
  } catch {
    // If the local model doesn't return clean JSON, fail open (pass) rather
    // than looping forever — but log the raw output so it's debuggable.
    logStep("Critic", "Could not parse critic output as JSON; defaulting to 'pass'.");
    logStep("Critic", `Raw output: ${raw.slice(0, 300)}`);
    return { verdict: "pass", issues: [] };
  }
}
