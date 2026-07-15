import fs from "node:fs/promises";
import pdfParse from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { CHUNK_SIZE, CHUNK_OVERLAP } from "../config.js";
import { log } from "../utils/logger.js";

/**
 * Loads a PDF from disk and returns its raw text plus basic metadata.
 */
export async function loadPdf(filePath) {
  const buffer = await fs.readFile(filePath);
  const data = await pdfParse(buffer);
  return {
    text: data.text,
    numPages: data.numpages,
    info: data.info,
  };
}

/**
 * Splits raw text into overlapping chunks suitable for embedding, using
 * LangChain's recursive splitter (tries to break on paragraph/sentence
 * boundaries first, falling back to hard splits).
 * Returns LangChain `Document` objects with a chunkIndex attached so later
 * stages (retrieval, critic fact-checking) can reference chunks by id.
 */
export async function splitText(text, { chunkSize = CHUNK_SIZE, chunkOverlap = CHUNK_OVERLAP } = {}) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
    separators: ["\n\n", "\n", ". ", " ", ""],
  });
  const docs = await splitter.createDocuments([text]);
  return docs.map((doc, i) => {
    doc.metadata = { ...doc.metadata, chunkIndex: i };
    return doc;
  });
}

/**
 * High-level ingestion helper used directly by the orchestration graph.
 */
export async function ingestPdf(filePath) {
  log(`Reading PDF: ${filePath}`);
  const { text, numPages } = await loadPdf(filePath);
  log(`Parsed ${numPages} pages, ${text.length} characters. Splitting into chunks...`);
  const chunks = await splitText(text);
  log(`Created ${chunks.length} chunks (chunkSize=${CHUNK_SIZE}, overlap=${CHUNK_OVERLAP}).`);
  return { chunks, numPages };
}

/**
 * The same capability exposed as a LangChain Tool object, so it can be
 * handed to an autonomous tool-calling agent instead of (or in addition to)
 * being called directly by the deterministic graph node above.
 */
export const pdfExtractionTool = tool(
  async ({ filePath }) => {
    const { text, numPages } = await loadPdf(filePath);
    return JSON.stringify({ numPages, preview: text.slice(0, 500) });
  },
  {
    name: "pdf_extract",
    description: "Extract raw text and page count from a PDF file on disk, given its file path.",
    schema: z.object({
      filePath: z.string().describe("Absolute or relative path to the PDF file"),
    }),
  }
);
