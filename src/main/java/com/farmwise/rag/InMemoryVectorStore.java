package com.farmwise.rag;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A simple in-memory vector store using cosine similarity. It is the Java port's
 * default backend (ChromaDB has no official Java client); because it only depends
 * on the {@link VectorStore} interface, Session 2 can drop in a pgvector-backed
 * store without touching {@link Assistant}.
 */
public final class InMemoryVectorStore implements VectorStore {

    private record Entry(String id, String document, double[] embedding, Map<String, Object> metadata) {}

    private final Map<String, Entry> entries = new LinkedHashMap<>();

    @Override
    public void add(List<String> ids, List<String> documents,
                    List<double[]> embeddings, List<Map<String, Object>> metadatas) {
        for (int i = 0; i < ids.size(); i++) {
            entries.put(ids.get(i),
                    new Entry(ids.get(i), documents.get(i), embeddings.get(i), metadatas.get(i)));
        }
    }

    @Override
    public List<SearchResult> search(double[] queryEmbedding, int topK) {
        List<SearchResult> scored = new ArrayList<>();
        for (Entry e : entries.values()) {
            double sim = Embeddings.cosine(queryEmbedding, e.embedding());
            double score = Math.max(0.0, sim); // clamp to [0, 1]
            scored.add(new SearchResult(e.id(), e.document(), e.metadata(), score));
        }
        scored.sort(Comparator.comparingDouble(SearchResult::score).reversed());
        return new ArrayList<>(scored.subList(0, Math.min(topK, scored.size())));
    }

    @Override
    public int count() {
        return entries.size();
    }

    @Override
    public void reset() {
        entries.clear();
    }
}
