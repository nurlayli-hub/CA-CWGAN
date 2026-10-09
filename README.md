# Data

The datasets are public and are not redistributed here. Put the files in this
folder (or wherever `DATA_DIR` points) before running notebook 01.

| Dataset | Source | File expected by notebook 01 |
|---|---|---|
| CDC Diabetes Health Indicators (3-class target `Diabetes_012`, 253,680 rows, 21 features) | https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators — the 3-class version is `diabetes_012_health_indicators_BRFSS2015.csv` of the same author's release on Kaggle (alexteboul/diabetes-health-indicators-dataset) | `diabetes_012_health_indicators_BRFSS2015.csv` |
| UCI Yeast | https://archive.ics.uci.edu/dataset/110/yeast | `yeast.data` (converted to `yeast.csv`) |
| UCI Ecoli | https://archive.ics.uci.edu/dataset/39/ecoli | `ecoli.data` (converted to `ecoli.csv`) |
| UCI Page Blocks | https://archive.ics.uci.edu/dataset/78/page+blocks+classification | `page-blocks.data.Z` (converted to `page-blocks.csv`) |

Notebook 01 converts the three raw UCI files to CSV the first time it runs (the
`.Z` file is decompressed with `gzip`, available in Colab and on Linux/macOS). If the
converted CSV is already present it is used directly. If your diabetes file has a
different name, edit `PRESETS` in notebooks 01 and 08.

Files produced by the pipeline (`df_train_*.csv`, `df_test_*.csv`, `*_pool.csv`,
`smotenc_*.csv`, results) are written to the same folder; see `PIPELINE.md`.
