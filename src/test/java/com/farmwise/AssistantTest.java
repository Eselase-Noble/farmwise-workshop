package com.farmwise;

import com.farmwise.llm.OllamaClient;
import com.farmwise.rag.Assistant;
import com.farmwise.rag.Embeddings;
import com.farmwise.rag.InMemoryVectorStore;
import com.farmwise.rag.RagAnswer;
import com.farmwise.rag.SearchResult;
import com.farmwise.rag.VectorStore;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Offline tests for the assistant: a fake embedder + a stub LLM client. */
class AssistantTest {

    private static VectorStore populatedStore() {
        VectorStore store = new InMemoryVectorStore();
        store.add(
                List.of("maize.md::0", "soil.md::1"),
                List.of("Maize responds strongly to nitrogen.", "Lime raises soil pH over time."),
                List.of(new double[] {1, 0, 0}, new double[] {0, 1, 0}),
                List.of(Map.of("source", "maize.md", "chunk_index", 0),
                        Map.of("source", "soil.md", "chunk_index", 1)));
        return store;
    }

    private static Embeddings fixedEmbedder() {
        return new Embeddings(null, null) {
            @Override public double[] embedQuery(String text) { return new double[] {1, 0, 0}; }
        };
    }

    private static OllamaClient stubClient(String reply) {
        return new OllamaClient() {
            @Override public String generate(String prompt, String system, String model,
                                             Integer maxTokens, Double temperature) {
                return reply;
            }
        };
    }

    @Test
    void buildPromptContainsContextAndQuestion() {
        List<SearchResult> results = List.of(
                new SearchResult("maize.md::0", "Maize needs nitrogen.",
                        Map.of("source", "maize.md"), 0.82));
        String prompt = Assistant.buildPrompt("How much nitrogen for maize?", results);
        assertTrue(prompt.contains("[1]"));
        assertTrue(prompt.contains("maize.md"));
        assertTrue(prompt.contains("How much nitrogen for maize?"));
    }

    @Test
    void ragReturnsSources() {
        Assistant a = new Assistant(populatedStore(), fixedEmbedder(),
                stubClient("Apply nitrogen; see maize guide."));
        RagAnswer answer = a.ask("How much nitrogen for maize?", 2);
        assertTrue(answer.usedContext());
        assertEquals(2, answer.sources().size());
        assertEquals("maize.md", answer.sources().get(0).source()); // nearest to [1,0,0]
    }

    @Test
    void bareLlmHasNoSources() {
        Assistant a = new Assistant(populatedStore(), fixedEmbedder(),
                stubClient("Generic answer."));
        RagAnswer answer = a.askWithoutRag("What is maize?");
        assertFalse(answer.usedContext());
        assertTrue(answer.sources().isEmpty());
        assertEquals("Generic answer.", answer.answer());
    }
}
