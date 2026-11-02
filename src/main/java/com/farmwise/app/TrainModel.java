package com.farmwise.app;

import com.farmwise.Config;
import com.farmwise.ml.CropData;
import com.farmwise.ml.Trainer;
import weka.core.Instances;

/**
 * Train and save the crop-recommendation model.
 * Run: {@code mvn exec:java -Dexec.mainClass=com.farmwise.app.TrainModel}
 */
public final class TrainModel {
    public static void main(String[] args) {
        System.out.println("Training crop-recommendation model...\n");
        try {
            Instances data = CropData.load(Config.CROP_DATASET_PATH);
            Trainer.Result r = Trainer.train(data);
            Trainer.save(r);

            System.out.printf("RandomForest accuracy : %.4f%n", r.accuracy());
            System.out.printf("LogReg baseline       : %.4f%n", r.baselineAccuracy());
            System.out.println("Classes               : " + r.classes().size());
            System.out.printf("Train / test rows     : %d / %d%n", r.nTrain(), r.nTest());
            System.out.println("\nSaved model    -> " + Config.MODEL_PATH);
            System.out.println("Saved metadata -> " + Config.MODEL_METADATA_PATH);
        } catch (CropData.DatasetNotFoundException e) {
            System.err.println("ERROR: " + e.getMessage());
            System.exit(1);
        }
    }
}
