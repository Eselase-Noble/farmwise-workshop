package com.farmwise;

import com.farmwise.llm.OllamaClient;
import com.farmwise.ml.CropFeatures;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class MlAndClientTest {

    @Test
    void cropFeaturesOrderAndFromMap() {
        CropFeatures f = new CropFeatures(1, 2, 3, 4, 5, 6, 7);
        assertArrayEquals(new double[] {1, 2, 3, 4, 5, 6, 7}, f.toArray());

        CropFeatures g = CropFeatures.fromMap(Map.of(
                "N", 1, "P", 2, "K", 3, "temperature", 4,
                "humidity", 5, "ph", 6, "rainfall", 7, "extra", 99));
        assertEquals(f, g); // extra key ignored
    }

    @Test
    void isRunningFalseWhenServerDown() {
        // Port 1 is never an Ollama server -> connection refused -> false, no throw.
        OllamaClient client = new OllamaClient("http://localhost:1");
        assertFalse(client.isRunning());
    }
}
