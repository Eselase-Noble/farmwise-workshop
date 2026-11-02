package com.farmwise;

import com.farmwise.rag.InMemoryVectorStore;
import com.farmwise.rag.SearchResult;
import com.farmwise.rag.VectorStore;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class VectorStoreTest {

    @Test
    void addSearchCountReset() {
        VectorStore store = new InMemoryVectorStore();
        store.add(
                List.of("a::0", "b::0", "c::0"),
                List.of("maize needs nitrogen", "tomato early blight", "soil ph lime"),
                List.of(new double[] {1, 0, 0}, new double[] {0, 1, 0}, new double[] {0, 0, 1}),
                List.of(Map.of("source", "maize.md"), Map.of("source", "tomato.md"),
                        Map.of("source", "soil.md")));
        assertEquals(3, store.count());

        List<SearchResult> results = store.search(new double[] {0.9, 0.1, 0.0}, 2);
        assertEquals(2, results.size());
        assertEquals("maize.md", results.get(0).metadata().get("source")); // nearest to [1,0,0]
        assertTrue(results.get(0).score() >= results.get(1).score());      // sorted
        assertTrue(results.get(0).score() >= 0.0 && results.get(0).score() <= 1.0);

        store.reset();
        assertEquals(0, store.count());
    }
}
