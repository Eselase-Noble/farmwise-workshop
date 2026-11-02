# data/raw/ — Crop Recommendation dataset

The crop-recommendation model is trained on the public **Crop Recommendation**
dataset. The CSV is **not committed** to git (see `.gitignore`); download it and
place it here before training.

## Expected file

```
data/raw/Crop_recommendation.csv
```

(Override the location with `FARMWISE_CROP_DATASET` if needed.)

## Expected columns

| column        | meaning                          |
|---------------|----------------------------------|
| `N`           | nitrogen content in soil         |
| `P`           | phosphorus content in soil       |
| `K`           | potassium content in soil        |
| `temperature` | °C                               |
| `humidity`    | %                                |
| `ph`          | soil pH                          |
| `rainfall`    | mm                               |
| `label`       | recommended crop (target)        |

~2,200 rows, 22 crop classes.

## Where to download

- **Kaggle:** https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
  (free login; file is `Crop_recommendation.csv`)
- **Mirror (no login):**
  https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-raw/crop_recommendation.csv

`farmwise.ml.data.load_crop_dataset()` validates the columns and raises a clear
error if the file is missing.

## ⚠️ Provenance note (for the Responsible-AI segment)

The dataset's collection **location and methodology are undocumented**. It is
widely understood to reflect **Indian** agronomic conditions and has **not** been
validated for Ghana. This is exactly the point explored in
`notebooks/03_responsible_ai.ipynb`.
