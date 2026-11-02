package com.farmwise;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Central configuration, mirroring the Python {@code farmwise.config} module.
 *
 * <p>Every value can be overridden with an environment variable so a laptop demo
 * and a server deployment share one code path. Import these constants rather than
 * hard-coding paths or model names.
 */
public final class Config {

    private Config() {}

    /** Project root = current working directory (run Maven from the repo root). */
    public static final Path PROJECT_ROOT = Paths.get("").toAbsolutePath();

    public static final Path DATA_DIR   = envPath("FARMWISE_DATA_DIR",   PROJECT_ROOT.resolve("data"));
    public static final Path RAW_DATA_DIR = envPath("FARMWISE_RAW_DATA_DIR", DATA_DIR.resolve("raw"));
    public static final Path GUIDES_DIR = envPath("FARMWISE_GUIDES_DIR", DATA_DIR.resolve("guides"));
    public static final Path MODELS_DIR = envPath("FARMWISE_MODELS_DIR", PROJECT_ROOT.resolve("models"));

    public static final Path CROP_DATASET_PATH =
            envPath("FARMWISE_CROP_DATASET", RAW_DATA_DIR.resolve("Crop_recommendation.csv"));
    public static final Path MODEL_PATH =
            envPath("FARMWISE_MODEL_PATH", MODELS_DIR.resolve("crop_model.model"));
    public static final Path MODEL_METADATA_PATH =
            envPath("FARMWISE_MODEL_METADATA", MODELS_DIR.resolve("crop_model_metadata.json"));

    // Ollama / models
    public static final String OLLAMA_HOST     = env("OLLAMA_HOST", "http://localhost:11434");
    public static final String LLM_MODEL       = env("FARMWISE_LLM_MODEL", "llama3.2:3b");
    public static final String EMBEDDING_MODEL = env("FARMWISE_EMBEDDING_MODEL", "nomic-embed-text");

    // RAG parameters (small for fast laptop responses)
    public static final int CHUNK_SIZE    = envInt("FARMWISE_CHUNK_SIZE", 800);
    public static final int CHUNK_OVERLAP = envInt("FARMWISE_CHUNK_OVERLAP", 150);
    public static final int TOP_K         = envInt("FARMWISE_TOP_K", 3);
    public static final int MAX_TOKENS    = envInt("FARMWISE_MAX_TOKENS", 256);
    public static final double LLM_TEMPERATURE = envDouble("FARMWISE_LLM_TEMPERATURE", 0.2);

    public static final String DATASET_SOURCE = env(
            "FARMWISE_DATASET_SOURCE",
            "Kaggle 'Crop Recommendation' dataset (Atharva Ingle). Collection location "
            + "and methodology are undocumented; widely understood to reflect Indian "
            + "agronomic conditions - NOT validated for Ghana.");

    // --- helpers ---------------------------------------------------------
    private static String env(String name, String def) {
        String v = System.getenv(name);
        return (v == null || v.isBlank()) ? def : v;
    }

    private static Path envPath(String name, Path def) {
        String v = System.getenv(name);
        return (v == null || v.isBlank()) ? def : Paths.get(v);
    }

    private static int envInt(String name, int def) {
        String v = System.getenv(name);
        return (v == null || v.isBlank()) ? def : Integer.parseInt(v.trim());
    }

    private static double envDouble(String name, double def) {
        String v = System.getenv(name);
        return (v == null || v.isBlank()) ? def : Double.parseDouble(v.trim());
    }
}
