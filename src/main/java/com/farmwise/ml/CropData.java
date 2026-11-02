package com.farmwise.ml;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Load and validate the Crop Recommendation dataset into Weka {@link Instances}.
 * Mirrors Python's {@code farmwise.ml.data}.
 */
public final class CropData {

    public static final String[] FEATURE_NAMES =
            {"N", "P", "K", "temperature", "humidity", "ph", "rainfall"};
    public static final String TARGET = "label";

    private CropData() {}

    /** Thrown when the dataset file is missing. */
    public static class DatasetNotFoundException extends RuntimeException {
        public DatasetNotFoundException(String m) { super(m); }
    }

    /** Thrown when the dataset is present but malformed. */
    public static class DatasetValidationException extends RuntimeException {
        public DatasetValidationException(String m) { super(m); }
    }

    /** Load the CSV, set the class column to {@code label}, and validate columns. */
    public static Instances load(Path csvPath) {
        if (!Files.exists(csvPath)) {
            throw new DatasetNotFoundException(
                    "Crop dataset not found at: " + csvPath
                    + "\nDownload the public 'Crop Recommendation' dataset into data/raw/ "
                    + "(see data/raw/README.md). Real data is required; do not fake it.");
        }
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(csvPath.toString()));
            Instances data = loader.getDataSet();
            validate(data);
            data.setClassIndex(data.attribute(TARGET).index());
            return data;
        } catch (IOException e) {
            throw new DatasetValidationException("Could not read dataset: " + e.getMessage());
        }
    }

    private static void validate(Instances data) {
        for (String col : FEATURE_NAMES) {
            Attribute a = data.attribute(col);
            if (a == null) {
                throw new DatasetValidationException("Dataset missing required column: " + col);
            }
            if (!a.isNumeric()) {
                throw new DatasetValidationException("Feature column must be numeric: " + col);
            }
        }
        if (data.attribute(TARGET) == null) {
            throw new DatasetValidationException("Dataset missing target column: " + TARGET);
        }
    }

    /** Per-feature observed [min, max], used to flag out-of-distribution inputs. */
    public static Map<String, double[]> featureRanges(Instances data) {
        Map<String, double[]> ranges = new LinkedHashMap<>();
        for (String col : FEATURE_NAMES) {
            int idx = data.attribute(col).index();
            var stats = data.attributeStats(idx).numericStats;
            ranges.put(col, new double[] {stats.min, stats.max});
        }
        return ranges;
    }
}
