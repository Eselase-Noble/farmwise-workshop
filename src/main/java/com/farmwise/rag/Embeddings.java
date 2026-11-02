package com.farmwise.rag;

import com.farmwise.llm.OllamaClient;

import java.util.ArrayList;
import java.util.List;

/**
 * Turn text into vectors via an Ollama embedding model (mirrors Python's
 * {@code farmwise.rag.embeddings}). Vectors are computed here and handed to the
 * vector store explicitly, keeping the store backend-agnostic.
 */
public class Embeddings {

    private final OllamaClient client;
    private final String model;

    public Embeddings() { this(new OllamaClient(), null); }

    public Embeddings(OllamaClient client, String model) {
        this.client = client;
        this.model = model;
    }

    public List<double[]> embedTexts(List<String> texts) {
        List<double[]> out = new ArrayList<>(texts.size());
        for (String t : texts) out.add(client.embed(t, model));
        return out;
    }

    public double[] embedQuery(String text) {
        return client.embed(text, model);
    }

    /** Cosine similarity in [-1, 1]; 1.0 means identical direction. */
    public static double cosine(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0.0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}
