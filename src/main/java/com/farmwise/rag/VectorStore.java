package com.farmwise.rag;

import java.util.List;
import java.util.Map;

/**
 * Minimal, backend-agnostic vector store interface (mirrors Python's
 * {@code VectorStore} ABC): {@code add / search / count / reset}.
 *
 * <p>The assistant depends only on this interface, so a PostgreSQL + pgvector
 * implementation can be added without changing the assistant — exactly the
 * Session 2 extension point.
 */
public interface VectorStore {

    /** Insert (or upsert) chunks with their precomputed embeddings. */
    void add(List<String> ids,
             List<String> documents,
             List<double[]> embeddings,
             List<Map<String, Object>> metadatas);

    /** Return the {@code topK} most similar chunks to the query embedding. */
    List<SearchResult> search(double[] queryEmbedding, int topK);

    /** How many chunks are stored. */
    int count();

    /** Remove everything (used when rebuilding the index). */
    void reset();
}
