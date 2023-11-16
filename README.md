## Directory structure

```nohighlight
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
└── src                <- Source code for use in this project.
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    └── models         <- Scripts to train models and then use trained models to make
        │                 predictions
        ├── predict_model.py
        └── train_model.py
```
