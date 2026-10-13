# expandingUSDatasets

## How to Run

```
python train_test_on_all_datasets_and_all_models.py
```

- `split_data.py` — Splits the dataset into training and test sets.
- `dataset.py` — Loads the data and performs data preprocessing.
- `syntheticImage.py` — Given a tumor image, a tumor location mask, and a normal image, generates a synthetic image along with its tumor location mask.
- `syntheticStrategy2.py` — Generates synthetic images according to Strategy 2, using different fusion pixel widths.
