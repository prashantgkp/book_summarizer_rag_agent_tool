import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { OLLAMA_BASE_URL, OLLAMA_API_KEY, CHAT_MODEL, EMBED_MODEL } from "./config.js";

/**
 * Creates a chat model instance pointed at the local Ollama server through
 * its OpenAI-compatible /v1 endpoint. Every agent that needs to "think"
 * calls this factory rather than instantiating ChatOpenAI directly, so the
 * routing/config stays in one place.
 *
 * Equivalent to the raw client shown in the project brief:
 *   const ollama = new OpenAI({ baseURL: "http://localhost:11434/v1", apiKey: "ollama" });
 * LangChain's ChatOpenAI wraps the same OpenAI SDK under the hood.
 */
export function makeChatModel({ temperature = 0.2 } = {}) {
  return new ChatOpenAI({
    model: CHAT_MODEL,
    temperature,
    apiKey: OLLAMA_API_KEY,
    configuration: {
      baseURL: OLLAMA_BASE_URL,
    },
  });
}

/**
 * Embeddings client, also routed through Ollama's OpenAI-compatible API
 * (Ollama exposes /v1/embeddings for pulled embedding models such as
 * nomic-embed-text or mxbai-embed-large).
 */
export const embeddings = new OpenAIEmbeddings({
  model: EMBED_MODEL,
  apiKey: OLLAMA_API_KEY,
  configuration: {
    baseURL: OLLAMA_BASE_URL,
  },
});
