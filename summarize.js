#!/usr/bin/env node
import path from "node:path";
import fs from "node:fs/promises";
import { runBookSummaryPipeline } from "./src/graph/orchestrator.js";
import { log } from "./src/utils/logger.js";

function parseArgs(argv) {
  const args = { format: "markdown", out: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--file" || a === "-f") args.file = argv[++i];
    else if (a === "--format") args.format = argv[++i];
    else if (a === "--out" || a === "-o") args.out = argv[++i];
    else if (a === "--help" || a === "-h") args.help = true;
  }
  return args;
}

function printHelp() {
  console.log(`
Usage: node summarize.js --file <path-to-pdf> [options]

Options:
  -f, --file <path>     Path to the PDF file to summarize (required)
      --format <type>   Output format: "markdown" (default) or "json"
  -o, --out <path>      Write output to a file instead of stdout
  -h, --help            Show this help message

Environment variables (see .env.example):
  OLLAMA_BASE_URL       Default: http://localhost:11434/v1
  OLLAMA_CHAT_MODEL     Default: llama3.1
  OLLAMA_EMBED_MODEL    Default: nomic-embed-text
`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.file) {
    printHelp();
    process.exit(args.help ? 0 : 1);
  }

  const filePath = path.resolve(args.file);
  log(`Starting multi-agent RAG summarization pipeline for: ${filePath}`);
  log("Agents: Retriever -> Summarizer (map) -> Synthesizer (reduce) -> Critic (QA)");

  const start = Date.now();
  const result = await runBookSummaryPipeline(filePath);
  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  log(`Pipeline complete in ${elapsed}s.`);

  const output =
    args.format === "json"
      ? JSON.stringify(
          {
            file: filePath,
            numPages: result.numPages,
            numChunks: result.chunks.length,
            numSections: result.sectionSummaries.length,
            critique: result.critique,
            revisionCount: result.revisionCount,
            summary: result.finalSummary,
          },
          null,
          2
        )
      : result.finalSummary;

  if (args.out) {
    await fs.writeFile(args.out, output, "utf-8");
    log(`Wrote output to ${args.out}`);
  } else {
    console.log("\n" + "=".repeat(80) + "\n");
    console.log(output);
  }
}

main().catch((err) => {
  console.error("Pipeline failed:", err);
  process.exit(1);
});
