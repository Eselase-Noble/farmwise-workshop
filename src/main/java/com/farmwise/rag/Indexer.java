package com.farmwise.rag;

import com.farmwise.Config;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Ingest guides into a vector store: load -> chunk -> embed -> store.
 * Mirrors Python's {@code farmwise.rag.indexer}; depends only on {@link VectorStore}.
 */
public final class Indexer {

    private Indexer() {}

    public record IndexReport(int documents, int chunks, List<String> sources) {}

    public static IndexReport indexGuides(VectorStore store, Embeddings embeddings) {
        return indexGuides(store, embeddings, Config.GUIDES_DIR,
                Config.CHUNK_SIZE, Config.CHUNK_OVERLAP, true);
    }

    public static IndexReport indexGuides(VectorStore store, Embeddings embeddings,
                                          Path directory, int chunkSize, int overlap, boolean reset) {
        List<Document> docs = Loader.loadGuides(directory);
        List<Chunk> chunks = Chunker.chunkDocuments(docs, chunkSize, overlap);
        if (chunks.isEmpty()) {
            throw new IllegalStateException(
                    "No chunks produced from " + directory + " (any .pdf/.txt/.md guides there?)");
        }
        if (reset) store.reset();

        List<String> texts = new ArrayList<>();
        List<String> ids = new ArrayList<>();
        List<Map<String, Object>> metas = new ArrayList<>();
        List<String> sources = new ArrayList<>();
        for (Chunk c : chunks) {
            texts.add(c.text());
            ids.add(c.source() + "::" + c.chunkIndex());
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("source", c.source());
            m.put("chunk_index", c.chunkIndex());
            metas.add(m);
        }
        for (Document d : docs) sources.add(d.source());

        List<double[]> vectors = embeddings.embedTexts(texts);
        store.add(ids, texts, vectors, metas);

        return new IndexReport(docs.size(), chunks.size(), sources);
    }
}
