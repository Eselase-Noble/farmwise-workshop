package com.farmwise;

import com.farmwise.rag.Chunk;
import com.farmwise.rag.Chunker;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ChunkerTest {

    @Test
    void overlapAndIndexing() {
        String text = "word ".repeat(400); // 2000 chars
        List<Chunk> chunks = Chunker.chunkText(text, "x.md", 800, 150);
        assertTrue(chunks.size() >= 2);
        for (int i = 0; i < chunks.size(); i++) {
            assertEquals(i, chunks.get(i).chunkIndex());
            assertEquals("x.md", chunks.get(i).source());
            assertTrue(chunks.get(i).text().length() <= 800);
        }
        // consecutive chunks overlap in the source (start steps back)
        assertTrue(chunks.get(1).start() < chunks.get(0).end());
    }

    @Test
    void shortTextIsOneChunk() {
        List<Chunk> chunks = Chunker.chunkText("just a short guide", "s.md", 800, 150);
        assertEquals(1, chunks.size());
        assertEquals("just a short guide", chunks.get(0).text());
    }

    @Test
    void rejectsBadOverlap() {
        assertThrows(IllegalArgumentException.class,
                () -> Chunker.chunkText("abc", "s.md", 100, 100));
    }
}
