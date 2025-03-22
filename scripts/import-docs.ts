import { readFile } from "node:fs/promises";
import { openai } from "@ai-sdk/openai";
import { LibSQLVector } from "@mastra/core/vector/libsql";
import { MDocument } from "@mastra/rag";
import { embedMany } from "ai";

const date = new Date().toISOString();
const vestorStoreIndexName = "sysdev_vocabulary";
const inputFilePath = "data/docs.json";

const vectorStore = new LibSQLVector({
  connectionUrl: `file:data/storage.db`,
});

await vectorStore.createIndex({
  indexName: vestorStoreIndexName,
  dimension: 1536, // "text-embedding-3-small" model dimension
});

console.log("Vector store setup");

const json = await readFile(inputFilePath, "utf-8");
const doc = MDocument.fromJSON(json);
console.log("MDocument created");

const chunks = await doc.chunk({
  strategy: "json",
  size: 512,
  overlap: 50,
  maxSize: 1000000,
  // TODO: 2025-03-22: The following error is unresolved.
  // RateLimitError: 429 Rate limit reached for gpt-4o in organization org-gfFHa5Ovdlgcmw83qEx0M6l5 on requests per min (RPM): Limit 500, Used 500, Requested 1. Please try again in 120ms.
  //
  // extract: {
  //   title: true,
  //   summary: true,
  //   keywords: true,
  // },
});

console.log("Chunks created: ", chunks.length);

const { embeddings } = await embedMany({
  model: openai.embedding("text-embedding-3-small"),
  values: chunks.map((chunk) => chunk.text),
});

console.log("Embeddings created: ", embeddings.length);

await vectorStore.upsert({
  indexName: vestorStoreIndexName,
  vectors: embeddings,
  metadata: chunks.map((chunk) => ({
    text: chunk.text,
    createdAt: date,
    // title: chunk.metadata.title,
    // summary: chunk.metadata.summary,
    // keywords: chunk.metadata.keywords,
  })),
});

console.log("Vectors stored");
