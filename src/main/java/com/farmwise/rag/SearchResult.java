package com.farmwise.rag;

import java.util.Map;

/**
 * One retrieved chunk and how well it matched the query.
 * {@code score} is similarity in [0, 1] (higher = more similar).
 */
public record SearchResult(
        String id,
        String document,
        Map<String, Object> metadata,
        double score) {}
