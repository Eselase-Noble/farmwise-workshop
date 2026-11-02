package com.farmwise.rag;

import com.farmwise.Config;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

/**
 * Load farming guides from disk (.pdf/.txt/.md). Mirrors Python's
 * {@code farmwise.rag.loader}; each file becomes one {@link Document} whose
 * source is the file name (used for citations).
 */
public final class Loader {

    private Loader() {}

    public static Document loadFile(Path path) {
        String name = path.getFileName().toString();
        String lower = name.toLowerCase(Locale.ROOT);
        try {
            String text;
            if (lower.endsWith(".pdf")) {
                text = readPdf(path);
            } else if (lower.endsWith(".txt") || lower.endsWith(".md")) {
                text = Files.readString(path);
            } else {
                throw new IllegalArgumentException(
                        "Unsupported file type for " + name + " (supported: .pdf, .txt, .md)");
            }
            return new Document(text.strip(), name);
        } catch (IOException e) {
            throw new RuntimeException("Could not read " + name + ": " + e.getMessage(), e);
        }
    }

    private static String readPdf(Path path) throws IOException {
        try (PDDocument doc = PDDocument.load(path.toFile())) {
            return new PDFTextStripper().getText(doc);
        }
    }

    /** Load every supported guide in the directory (skips README), sorted by name. */
    public static List<Document> loadGuides(Path directory) {
        if (!Files.isDirectory(directory)) {
            throw new RuntimeException("Guides directory not found: " + directory);
        }
        List<Document> docs = new ArrayList<>();
        try (Stream<Path> files = Files.list(directory)) {
            files.sorted().forEach(p -> {
                String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
                if (n.equals("readme.md")) return;
                if (!(n.endsWith(".pdf") || n.endsWith(".txt") || n.endsWith(".md"))) return;
                Document d = loadFile(p);
                if (d.text().isBlank()) {
                    System.out.println("[loader] WARNING: no text in " + p.getFileName() + ", skipping");
                } else {
                    docs.add(d);
                }
            });
        } catch (IOException e) {
            throw new RuntimeException("Could not list guides: " + e.getMessage(), e);
        }
        return docs;
    }

    public static List<Document> loadGuides() {
        return loadGuides(Config.GUIDES_DIR);
    }
}
