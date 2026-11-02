package com.farmwise.ml;

import java.util.Map;

/**
 * Soil and weather inputs for one farm (the model's seven features).
 * Typed input to {@link Predictor#predictCrop}, so a web layer can bind to it
 * directly (mirrors Python's {@code CropFeatures} dataclass).
 */
public record CropFeatures(
        double N, double P, double K,
        double temperature, double humidity, double ph, double rainfall) {

    /** Values in the canonical feature order used for training. */
    public double[] toArray() {
        return new double[] {N, P, K, temperature, humidity, ph, rainfall};
    }

    /** Build from a map keyed by feature name (ignores extra keys). */
    public static CropFeatures fromMap(Map<String, ? extends Number> m) {
        return new CropFeatures(
                m.get("N").doubleValue(), m.get("P").doubleValue(), m.get("K").doubleValue(),
                m.get("temperature").doubleValue(), m.get("humidity").doubleValue(),
                m.get("ph").doubleValue(), m.get("rainfall").doubleValue());
    }
}
