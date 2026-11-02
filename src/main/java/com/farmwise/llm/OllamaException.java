package com.farmwise.llm;

/** Problems talking to the local Ollama server. */
public class OllamaException extends RuntimeException {
    public OllamaException(String message) { super(message); }
    public OllamaException(String message, Throwable cause) { super(message, cause); }

    /** The Ollama server could not be reached. */
    public static class NotRunning extends OllamaException {
        public NotRunning(String host, Throwable cause) {
            super("Could not reach the Ollama server at " + host
                    + ".\nStart it with:  ollama serve"
                    + "\n(Verify with:  mvn exec:java -Dexec.mainClass=com.farmwise.app.CheckSetup)", cause);
        }
    }

    /** A requested model is not available locally. */
    public static class ModelNotFound extends OllamaException {
        public ModelNotFound(String model) {
            super("Ollama model '" + model + "' is not available.\nPull it with:  ollama pull " + model);
        }
    }
}
