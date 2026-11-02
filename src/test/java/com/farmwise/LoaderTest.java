package com.farmwise;

import com.farmwise.rag.Document;
import com.farmwise.rag.Loader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

class LoaderTest {

    @Test
    void loadsMarkdown(@TempDir Path dir) throws Exception {
        Path f = dir.resolve("a.md");
        Files.writeString(f, "# Title\nhello");
        Document doc = Loader.loadFile(f);
        assertEquals("a.md", doc.source());
        assertTrue(doc.text().contains("hello"));
    }

    @Test
    void rejectsUnsupportedType(@TempDir Path dir) throws Exception {
        Path f = dir.resolve("a.docx");
        Files.writeString(f, "nope");
        assertThrows(IllegalArgumentException.class, () -> Loader.loadFile(f));
    }

    @Test
    void loadGuidesSkipsReadme(@TempDir Path dir) throws Exception {
        Files.writeString(dir.resolve("README.md"), "readme");
        Files.writeString(dir.resolve("guide1.md"), "content one");
        Files.writeString(dir.resolve("guide2.txt"), "content two");
        List<Document> docs = Loader.loadGuides(dir);
        Set<String> sources = docs.stream().map(Document::source).collect(Collectors.toSet());
        assertEquals(Set.of("guide1.md", "guide2.txt"), sources);
    }
}
