package com.farmwise.app;

import com.farmwise.llm.OllamaClient;
import com.farmwise.ml.CropFeatures;
import com.farmwise.ml.CropPrediction;
import com.farmwise.ml.Predictor;
import com.farmwise.rag.Assistant;
import com.farmwise.rag.Embeddings;
import com.farmwise.rag.Indexer;
import com.farmwise.rag.InMemoryVectorStore;
import com.farmwise.rag.RagAnswer;
import com.farmwise.rag.VectorStore;

/**
 * End-to-end demonstration of both FarmWise features (the Java equivalent of the
 * notebooks' finale). Trains must have run first; Ollama must be up.
 *
 * Run: {@code mvn exec:java -Dexec.mainClass=com.farmwise.app.Demo}
 */
public final class Demo {
    public static void main(String[] args) {
        // --- Feature 1: crop recommendation ---
        System.out.println("=== Crop recommendation ===");
        CropFeatures farm = new CropFeatures(90, 42, 43, 21, 82, 6.5, 200);
        CropPrediction pred = Predictor.predictCrop(farm);
        System.out.println("Recommended : " + pred.crop());
        System.out.printf("Confidence  : %.3f%n", pred.confidence());
        System.out.print("Top 3       : ");
        pred.top3().forEach(s -> System.out.printf("%s=%.3f  ", s.crop(), s.probability()));
        System.out.println("\nOut of range: " + pred.outOfDistribution());

        // --- Feature 2: RAG farming assistant ---
        System.out.println("\n=== Farming Q&A (RAG) ===");
        OllamaClient client = new OllamaClient();
        Embeddings embeddings = new Embeddings();
        VectorStore store = new InMemoryVectorStore();

        Indexer.IndexReport report = Indexer.indexGuides(store, embeddings);
        System.out.printf("Indexed %d guide(s) -> %d chunk(s)%n", report.documents(), report.chunks());

        Assistant assistant = new Assistant(store, embeddings, client);

        String q1 = "What is the critical weed-free period for maize?";
        RagAnswer a1 = assistant.ask(q1);
        System.out.println("\nQ: " + q1);
        System.out.println("RAG: " + a1.answer());
        System.out.print("Sources: ");
        a1.sources().forEach(s -> System.out.print(s.source() + " "));

        String q2 = "How do I raise tilapia fish in a pond?"; // not in the guides
        RagAnswer a2 = assistant.ask(q2);
        System.out.println("\n\nQ: " + q2);
        System.out.println("RAG: " + a2.answer());
    }
}
