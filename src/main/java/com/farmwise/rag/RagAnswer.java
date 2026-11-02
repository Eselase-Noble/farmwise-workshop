package com.farmwise.rag;

import java.util.List;

/**
 * A typed answer from either assistant entry point (mirrors Python's
 * {@code RagAnswer}). {@code usedContext} is true for RAG, false for the bare LLM.
 */
public record RagAnswer(
        String question,
        String answer,
        boolean usedContext,
        List<SourceChunk> sources,
        String model) {

    /** A retrieved guide chunk cited in a RAG answer. */
    public record SourceChunk(String source, int chunkIndex, String text, double score) {}
}
