package com.farmwise.ml;

import com.farmwise.Config;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Train, evaluate, and save the crop model (mirrors Python's
 * {@code farmwise.ml.train}). The production model is a Weka RandomForest; a
 * Logistic Regression baseline is trained alongside for comparison.
 */
public final class Trainer {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private Trainer() {}

    /** Everything a training run produces. */
    public record Result(
            Classifier model,
            Instances header,          // empty structure, used to build predict-time instances
            double accuracy,
            double baselineAccuracy,
            List<String> classes,
            Map<String, double[]> featureRanges,
            String classDetails,
            String confusionMatrix,
            int nTrain,
            int nTest,
            int nTrees) {}

    public static Result train(Instances data) {
        return train(data, 200, 42, 0.8);
    }

    public static Result train(Instances data, int nTrees, long seed, double trainFraction) {
        try {
            Map<String, double[]> ranges = CropData.featureRanges(data);

            Instances shuffled = new Instances(data);
            shuffled.randomize(new Random(seed));
            int n = shuffled.numInstances();
            int trainSize = (int) Math.round(n * trainFraction);
            Instances train = new Instances(shuffled, 0, trainSize);
            Instances test = new Instances(shuffled, trainSize, n - trainSize);

            RandomForest rf = new RandomForest();
            rf.setNumIterations(nTrees);
            rf.setSeed((int) seed);
            rf.buildClassifier(train);

            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(rf, test);
            double accuracy = eval.pctCorrect() / 100.0;

            Logistic baseline = new Logistic();
            baseline.buildClassifier(train);
            Evaluation beval = new Evaluation(train);
            beval.evaluateModel(baseline, test);
            double baselineAccuracy = beval.pctCorrect() / 100.0;

            List<String> classes = new ArrayList<>();
            var classAttr = data.classAttribute();
            for (int i = 0; i < classAttr.numValues(); i++) classes.add(classAttr.value(i));

            Instances header = new Instances(train, 0); // structure only

            return new Result(rf, header, accuracy, baselineAccuracy, classes, ranges,
                    eval.toClassDetailsString("Per-class performance (test set):"),
                    eval.toMatrixString("Confusion matrix (test set):"),
                    trainSize, n - trainSize, nTrees);
        } catch (Exception e) {
            throw new RuntimeException("Training failed: " + e.getMessage(), e);
        }
    }

    /** Persist the model (+ header) and a JSON metadata sidecar. */
    public static void save(Result r) {
        save(r, Config.MODEL_PATH, Config.MODEL_METADATA_PATH);
    }

    public static void save(Result r, Path modelPath, Path metadataPath) {
        try {
            Files.createDirectories(modelPath.getParent());
            // Store model AND header together so the predictor can rebuild instances.
            SerializationHelper.writeAll(modelPath.toString(), new Object[] {r.model(), r.header()});
            Files.writeString(metadataPath, MAPPER.writerWithDefaultPrettyPrinter()
                    .writeValueAsString(metadata(r)));
        } catch (Exception e) {
            throw new RuntimeException("Could not save model: " + e.getMessage(), e);
        }
    }

    private static ObjectNode metadata(Result r) {
        ObjectNode md = MAPPER.createObjectNode();
        md.put("model_type", "RandomForest");
        ArrayNode feats = md.putArray("feature_names");
        for (String f : CropData.FEATURE_NAMES) feats.add(f);
        ArrayNode cls = md.putArray("classes");
        for (String c : r.classes()) cls.add(c);
        md.put("n_classes", r.classes().size());
        md.put("accuracy", r.accuracy());
        md.put("baseline_model", "LogisticRegression");
        md.put("baseline_accuracy", r.baselineAccuracy());
        md.put("n_train", r.nTrain());
        md.put("n_test", r.nTest());
        md.put("n_estimators", r.nTrees());
        ObjectNode ranges = md.putObject("feature_ranges");
        for (var e : r.featureRanges().entrySet()) {
            ObjectNode b = ranges.putObject(e.getKey());
            b.put("min", e.getValue()[0]);
            b.put("max", e.getValue()[1]);
        }
        md.put("training_date", Instant.now().toString());
        md.put("dataset_source", Config.DATASET_SOURCE);
        return md;
    }
}
