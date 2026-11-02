package com.farmwise.app;

import com.farmwise.Config;
import com.farmwise.llm.OllamaClient;
import com.farmwise.rag.Embeddings;
import com.farmwise.rag.Indexer;
import com.farmwise.rag.InMemoryVectorStore;
import com.farmwise.rag.VectorStore;

/**
 * Demonstrate the ingestion pipeline (load -> chunk -> embed -> store) and report
 * counts. The Java port uses an in-memory store, so this is illustrative; a
 * persistent store (pgvector) is the Session 2 extension.
 *
 * Run: {@code mvn exec:java -Dexec.mainClass=com.farmwise.app.BuildIndex}
 */
public final class BuildIndex {
    public static void main(String[] args) {
        OllamaClient client = new OllamaClient();
        if (!client.isRunning()) {
            System.err.println("ERROR: Ollama not reachable at " + Config.OLLAMA_HOST
                    + "\nStart it with: ollama serve");
            System.exit(1);
        }
        System.out.println("Guides dir : " + Config.GUIDES_DIR);
        System.out.println("Embedding  : " + Config.EMBEDDING_MODEL);
        System.out.printf("Chunk size : %d (overlap %d)%n%n", Config.CHUNK_SIZE, Config.CHUNK_OVERLAP);

        VectorStore store = new InMemoryVectorStore();
        Indexer.IndexReport report = Indexer.indexGuides(store, new Embeddings());

        System.out.printf("Indexed %d document(s) -> %d chunk(s)%n", report.documents(), report.chunks());
        report.sources().forEach(s -> System.out.println("  - " + s));
        System.out.println("\nChunks in store: " + store.count());
    }
}
