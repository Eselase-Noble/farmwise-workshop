package com.farmwise.rag;

/** One loaded guide: its full text and the file it came from. */
public record Document(String text, String source) {}
