package com.farmwise.rag;

import com.farmwise.Config;
import com.farmwise.llm.OllamaClient;

import java.util.ArrayList;
import java.util.List;

/**
 * The farming Q&amp;A assistant: two entry points, one answer shape (mirrors
 * Python's {@code farmwise.rag.assistant}).
 *
 * <ul>
 *   <li>{@link #askWithoutRag} — the bare LLM: answer from the model's own
 *       knowledge, no documents (can hallucinate).</li>
 *   <li>{@link #ask} — RAG: retrieve guide chunks, answer only from them, cite
 *       sources, and say "I don't know" when the guides don't cover it.</li>
 * </ul>
 */
public final class Assistant {

    public static final String PLAIN_SYSTEM_PROMPT =
            "You are FarmWise, a helpful assistant for farmers. "
            + "Answer concisely in plain language.";

    public static final String RAG_SYSTEM_PROMPT =
            "You are FarmWise, a helpful assistant for farmers. "
            + "Answer the question using ONLY the numbered context passages provided. "
            + "If the answer is not contained in the context, reply exactly: "
            + "\"I don't know based on the farming guides provided.\" "
            + "Do not use outside knowledge and do not guess. Keep the answer short, "
            + "and mention which guide(s) the answer came from.";

    private final VectorStore store;
    private final Embeddings embeddings;
    private final OllamaClient client;

    public Assistant(VectorStore store, Embeddings embeddings, OllamaClient client) {
        this.store = store;
        this.embeddings = embeddings;
        this.client = client;
    }

    /** Answer from the LLM's own knowledge — no documents, no retrieval. */
    public RagAnswer askWithoutRag(String question) {
        String answer = client.generate(question, PLAIN_SYSTEM_PROMPT, Config.LLM_MODEL, null, null);
        return new RagAnswer(question, answer, false, List.of(), Config.LLM_MODEL);
    }

    /** Answer with RAG: retrieve, ground, cite. */
    public RagAnswer ask(String question) {
        return ask(question, Config.TOP_K);
    }

    public RagAnswer ask(String question, int topK) {
        double[] queryVec = embeddings.embedQuery(question);
        List<SearchResult> results = store.search(queryVec, topK);

        String prompt = buildPrompt(question, results);
        String answer = client.generate(prompt, RAG_SYSTEM_PROMPT, Config.LLM_MODEL, null, null);

        List<RagAnswer.SourceChunk> sources = new ArrayList<>();
        for (SearchResult r : results) {
            Object idx = r.metadata().getOrDefault("chunk_index", -1);
            sources.add(new RagAnswer.SourceChunk(
                    String.valueOf(r.metadata().getOrDefault("source", "?")),
                    idx instanceof Number nb ? nb.intValue() : -1,
                    r.document(), r.score()));
        }
        return new RagAnswer(question, answer, true, sources, Config.LLM_MODEL);
    }

    /** Assemble the RAG prompt from retrieved chunks (printed in the demo). */
    public static String buildPrompt(String question, List<SearchResult> results) {
        StringBuilder ctx = new StringBuilder();
        int i = 1;
        for (SearchResult r : results) {
            String source = String.valueOf(r.metadata().getOrDefault("source", "?"));
            ctx.append("[").append(i++).append("] (from ").append(source).append(")\n")
               .append(r.document()).append("\n\n");
        }
        if (results.isEmpty()) ctx.append("(no context found)\n\n");
        return "Context passages:\n" + ctx
                + "Question: " + question + "\n\n"
                + "Answer (use only the context above):";
    }
}
