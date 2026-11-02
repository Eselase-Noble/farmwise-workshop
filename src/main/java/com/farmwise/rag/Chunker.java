package com.farmwise.rag;

import com.farmwise.Config;

import java.util.ArrayList;
import java.util.List;

/**
 * Split documents into overlapping character windows. Mirrors Python's
 * {@code farmwise.rag.chunker}: deliberately simple and explainable. The window
 * advances by {@code size - overlap}, and each chunk's end is nudged back to the
 * nearest whitespace so words aren't cut in half.
 */
public final class Chunker {

    private Chunker() {}

    public static List<Chunk> chunkText(String text, String source) {
        return chunkText(text, source, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP);
    }

    public static List<Chunk> chunkText(String text, String source, int size, int overlap) {
        if (size <= 0) throw new IllegalArgumentException("chunk size must be positive");
        if (overlap < 0 || overlap >= size) {
            throw new IllegalArgumentException("overlap must be >= 0 and < size");
        }
        List<Chunk> chunks = new ArrayList<>();
        String t = text.strip();
        if (t.isEmpty()) return chunks;

        int n = t.length();
        int start = 0;
        int index = 0;
        while (start < n) {
            int end = Math.min(start + size, n);
            if (end < n) {
                int window = Math.max(start + (int) (size * 0.85), start + 1);
                int space = t.lastIndexOf(' ', end);
                if (space >= window) end = space;
            }
            String piece = t.substring(start, end).strip();
            if (!piece.isEmpty()) {
                chunks.add(new Chunk(piece, source, index++, start, end));
            }
            if (end >= n) break;
            start = end - overlap;
        }
        return chunks;
    }

    public static List<Chunk> chunkDocuments(List<Document> docs, int size, int overlap) {
        List<Chunk> all = new ArrayList<>();
        for (Document d : docs) all.addAll(chunkText(d.text(), d.source(), size, overlap));
        return all;
    }

    public static List<Chunk> chunkDocuments(List<Document> docs) {
        return chunkDocuments(docs, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP);
    }
}
