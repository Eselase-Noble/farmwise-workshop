package com.farmwise.rag;

/** A slice of a document, with enough context to cite it later. */
public record Chunk(String text, String source, int chunkIndex, int start, int end) {}
