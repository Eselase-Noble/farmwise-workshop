package com.farmwise.ml;

import java.util.List;

/** The model's recommendation plus trust signals (mirrors Python's dataclass). */
public record CropPrediction(
        String crop,
        double confidence,
        List<CropScore> top3,
        boolean outOfDistribution,
        List<String> oodWarnings) {

    /** One candidate crop and its predicted probability. */
    public record CropScore(String crop, double probability) {}
}
