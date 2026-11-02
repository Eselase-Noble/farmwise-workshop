package com.farmwise.ml;

import com.farmwise.Config;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Load the saved crop model and make predictions (mirrors Python's
 * {@code farmwise.ml.predict}). Flags out-of-distribution inputs so a caller
 * knows when the recommendation is extrapolation rather than evidence.
 */
public final class Predictor {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final Classifier model;
    private final Instances header;
    private final Map<String, double[]> ranges;

    private static Predictor cached;

    public Predictor(Classifier model, Instances header, Map<String, double[]> ranges) {
        this.model = model;
        this.header = header;
        this.ranges = ranges;
    }

    public static class ModelNotFoundException extends RuntimeException {
        public ModelNotFoundException(String m) { super(m); }
    }

    /** Load from the configured default paths. */
    public static Predictor load() {
        return load(Config.MODEL_PATH, Config.MODEL_METADATA_PATH);
    }

    public static Predictor load(Path modelPath, Path metadataPath) {
        if (!Files.exists(modelPath)) {
            throw new ModelNotFoundException(
                    "Trained model not found at: " + modelPath
                    + "\nTrain it first:  mvn exec:java -Dexec.mainClass=com.farmwise.app.TrainModel");
        }
        try {
            Object[] parts = SerializationHelper.readAll(modelPath.toString());
            Classifier model = (Classifier) parts[0];
            Instances header = (Instances) parts[1];
            Map<String, double[]> ranges = readRanges(metadataPath);
            return new Predictor(model, header, ranges);
        } catch (Exception e) {
            throw new RuntimeException("Could not load model: " + e.getMessage(), e);
        }
    }

    private static Map<String, double[]> readRanges(Path metadataPath) throws Exception {
        Map<String, double[]> ranges = new LinkedHashMap<>();
        if (!Files.exists(metadataPath)) return ranges;
        JsonNode md = MAPPER.readTree(Files.readString(metadataPath));
        JsonNode fr = md.path("feature_ranges");
        for (String f : CropData.FEATURE_NAMES) {
            JsonNode b = fr.path(f);
            if (b.has("min") && b.has("max")) {
                ranges.put(f, new double[] {b.get("min").asDouble(), b.get("max").asDouble()});
            }
        }
        return ranges;
    }

    /** Convenience: predict with a lazily-loaded default model. */
    public static CropPrediction predictCrop(CropFeatures features) {
        if (cached == null) cached = load();
        return cached.predict(features);
    }

    public CropPrediction predict(CropFeatures features) {
        try {
            Instance inst = new DenseInstance(header.numAttributes());
            inst.setDataset(header);
            double[] values = features.toArray();
            for (int i = 0; i < CropData.FEATURE_NAMES.length; i++) {
                inst.setValue(header.attribute(CropData.FEATURE_NAMES[i]), values[i]);
            }
            inst.setClassMissing();

            double[] dist = model.distributionForInstance(inst);
            var classAttr = header.classAttribute();

            List<CropPrediction.CropScore> scores = new ArrayList<>();
            for (int i = 0; i < dist.length; i++) {
                scores.add(new CropPrediction.CropScore(classAttr.value(i), dist[i]));
            }
            scores.sort(Comparator.comparingDouble(CropPrediction.CropScore::probability).reversed());
            List<CropPrediction.CropScore> top3 = scores.subList(0, Math.min(3, scores.size()));

            List<String> warnings = outOfDistribution(features);
            return new CropPrediction(
                    top3.get(0).crop(), top3.get(0).probability(),
                    new ArrayList<>(top3), !warnings.isEmpty(), warnings);
        } catch (Exception e) {
            throw new RuntimeException("Prediction failed: " + e.getMessage(), e);
        }
    }

    private List<String> outOfDistribution(CropFeatures features) {
        List<String> warnings = new ArrayList<>();
        double[] values = features.toArray();
        for (int i = 0; i < CropData.FEATURE_NAMES.length; i++) {
            String name = CropData.FEATURE_NAMES[i];
            double[] b = ranges.get(name);
            if (b == null) continue;
            double v = values[i];
            if (v < b[0] || v > b[1]) {
                warnings.add(String.format("%s=%s is outside the training range [%.3f, %.3f]",
                        name, trim(v), b[0], b[1]));
            }
        }
        return warnings;
    }

    private static String trim(double v) {
        if (v == Math.rint(v)) return String.valueOf((long) v);
        return String.valueOf(v);
    }
}
