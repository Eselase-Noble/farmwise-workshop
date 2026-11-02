package com.farmwise.llm;

import com.farmwise.Config;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.ConnectException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;

/**
 * A thin wrapper around a local Ollama server, mirroring Python's
 * {@code farmwise.llm.client}. Only what the demo needs: generate, chat,
 * embeddings, plus health/model checks, with clear errors when Ollama is down
 * or a model is missing.
 */
public class OllamaClient {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5)).build();
    private final String host;

    public OllamaClient() { this(Config.OLLAMA_HOST); }
    public OllamaClient(String host) { this.host = host.replaceAll("/+$", ""); }

    /** True if the server answers a tags request. */
    public boolean isRunning() {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(host + "/api/tags"))
                    .timeout(Duration.ofSeconds(5)).GET().build();
            return http.send(req, HttpResponse.BodyHandlers.discarding()).statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    /** Names of locally available models. */
    public List<String> listModels() {
        JsonNode root = get("/api/tags");
        return root.path("models").findValuesAsText("name");
    }

    /** Generate a completion for a single prompt. */
    public String generate(String prompt, String system, String model, Integer maxTokens, Double temperature) {
        ObjectNode payload = MAPPER.createObjectNode();
        payload.put("model", model != null ? model : Config.LLM_MODEL);
        payload.put("prompt", prompt);
        payload.put("stream", false);
        ObjectNode options = payload.putObject("options");
        options.put("num_predict", maxTokens != null ? maxTokens : Config.MAX_TOKENS);
        options.put("temperature", temperature != null ? temperature : Config.LLM_TEMPERATURE);
        if (system != null) payload.put("system", system);

        JsonNode resp = post("/api/generate", payload);
        return resp.path("response").asText("").strip();
    }

    public String generate(String prompt, String system) {
        return generate(prompt, system, null, null, null);
    }

    /** Embed a single text into a vector. */
    public double[] embed(String text, String model) {
        ObjectNode payload = MAPPER.createObjectNode();
        payload.put("model", model != null ? model : Config.EMBEDDING_MODEL);
        payload.put("prompt", text);
        JsonNode resp = post("/api/embeddings", payload);
        ArrayNode arr = (ArrayNode) resp.path("embedding");
        double[] vec = new double[arr.size()];
        for (int i = 0; i < arr.size(); i++) vec[i] = arr.get(i).asDouble();
        return vec;
    }

    // --- HTTP plumbing ---------------------------------------------------
    private JsonNode get(String path) {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(host + path))
                    .timeout(Duration.ofSeconds(10)).GET().build();
            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            return MAPPER.readTree(resp.body());
        } catch (ConnectException e) {
            throw new OllamaException.NotRunning(host, e);
        } catch (Exception e) {
            if (isConnRefused(e)) throw new OllamaException.NotRunning(host, e);
            throw new OllamaException("Ollama GET " + path + " failed: " + e.getMessage(), e);
        }
    }

    private JsonNode post(String path, ObjectNode payload) {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(host + path))
                    .timeout(Duration.ofSeconds(120))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(payload)))
                    .build();
            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() == 404) {
                throw new OllamaException.ModelNotFound(payload.path("model").asText("?"));
            }
            JsonNode node = MAPPER.readTree(resp.body());
            if (node.hasNonNull("error")) {
                String err = node.get("error").asText();
                if (err.toLowerCase().contains("not found")) {
                    throw new OllamaException.ModelNotFound(payload.path("model").asText("?"));
                }
                throw new OllamaException("Ollama error: " + err);
            }
            return node;
        } catch (OllamaException e) {
            throw e;
        } catch (ConnectException e) {
            throw new OllamaException.NotRunning(host, e);
        } catch (Exception e) {
            if (isConnRefused(e)) throw new OllamaException.NotRunning(host, e);
            throw new OllamaException("Ollama POST " + path + " failed: " + e.getMessage(), e);
        }
    }

    private static boolean isConnRefused(Throwable e) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t instanceof ConnectException) return true;
            String m = t.getMessage();
            if (m != null && m.toLowerCase().contains("connection refused")) return true;
        }
        return false;
    }
}
