import { openai } from "@ai-sdk/openai";
import { LibSQLVector } from "@mastra/core/vector/libsql";
import { embed } from "ai";

const query = `Please explain the difference between a test item and a test case.`;
const vestorStoreIndexName = "sysdev_vocabulary";

const vectorStore = new LibSQLVector({
  connectionUrl: `file:data/storage.db`,
});

const { embedding } = await embed({
  model: openai.embedding("text-embedding-3-small"),
  value: query,
});

const searchResults = await vectorStore.query({
  indexName: vestorStoreIndexName,
  queryVector: embedding,
  topK: 20,
});

console.log("Search results: ", searchResults);
