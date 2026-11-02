package com.farmwise.app;

import com.farmwise.Config;
import com.farmwise.llm.OllamaClient;

import java.nio.file.Files;
import java.util.List;

/**
 * Pre-flight check for the Java demo.
 * Run: {@code mvn exec:java -Dexec.mainClass=com.farmwise.app.CheckSetup}
 */
public final class CheckSetup {

    private static final String OK = "  [ OK ]";
    private static final String FAIL = "  [FAIL]";
    private static final String WARN = "  [WARN]";

    public static void main(String[] args) {
        System.out.println("FarmWise (Java) setup check\n");
        boolean all = true;

        OllamaClient client = new OllamaClient();
        boolean ollamaUp = client.isRunning();
        if (ollamaUp) {
            System.out.println(OK + " Ollama server reachable at " + Config.OLLAMA_HOST);
        } else {
            System.out.println(FAIL + " Ollama not reachable at " + Config.OLLAMA_HOST);
            System.out.println("        Fix: ollama serve");
            all = false;
        }

        if (ollamaUp) {
            List<String> models = client.listModels();
            for (String m : new String[] {Config.LLM_MODEL, Config.EMBEDDING_MODEL}) {
                boolean present = models.stream().anyMatch(
                        x -> x.equals(m) || x.split(":")[0].equals(m.split(":")[0]));
                if (present) {
                    System.out.println(OK + " model present: " + m);
                } else {
                    System.out.println(FAIL + " model missing: " + m);
                    System.out.println("        Fix: ollama pull " + m);
                    all = false;
                }
            }
        } else {
            System.out.println(WARN + " skipping model check (Ollama not running)");
        }

        all &= checkPath("crop dataset", Config.CROP_DATASET_PATH,
                "download it into data/raw/ (see data/raw/README.md)");
        all &= checkPath("trained model", Config.MODEL_PATH,
                "mvn exec:java -Dexec.mainClass=com.farmwise.app.TrainModel");

        System.out.println();
        if (all) {
            System.out.println("All checks passed — ready to demo.");
        } else {
            System.out.println("Some checks failed — see the fixes above.");
            System.exit(1);
        }
    }

    private static boolean checkPath(String label, java.nio.file.Path path, String fix) {
        if (Files.exists(path)) {
            System.out.println(OK + " " + label + ": " + path);
            return true;
        }
        System.out.println(FAIL + " " + label + " missing: " + path);
        System.out.println("        Fix: " + fix);
        return false;
    }
}
