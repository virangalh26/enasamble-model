# Dataset description: RECON-SL

**Name:** RECON-SL, the Residential Electricity Consumption Dataset for Sri Lanka
**Publisher:** LIRNEasia, on IEEE DataPort (Algama et al., 2025), doi:[10.21227/n1dk-q860](https://dx.doi.org/10.21227/n1dk-q860)
**Mirror used:** [Kaggle: lirneasia/sri-lankan-residential-electricity-consumption](https://www.kaggle.com/datasets/lirneasia/sri-lankan-residential-electricity-consumption)

## How it was obtained

The dataset is public. It is downloaded by a script, not by hand:

```bash
python -m src.download --out data/raw/recon_sl.zip
```

The download is about 1.2 GB, so it is not included in the submission. The raw files and every
processed version are tracked with DVC (`dvc.yaml`, `dvc.lock`) in Azure Blob Storage, and
`dvc repro` rebuilds the whole pipeline from the download.

## What it contains

* 15-minute smart-meter readings from LECO residential customers in the Moratuwa area, October 2023 to
  December 2024, about 45.6 million readings.
* Cumulative import and export energy registers, plus power, power factor and meter metadata.
* Customers appear only as anonymous IDs (for example `ID0183`). There are no names, addresses or exact
  locations.

## How it is used in this project

| Item | Value |
|---|---|
| Unit of analysis | one household-day (24 hourly import values) |
| Complete household-days after cleaning | 452,811 |
| Households / with solar | 2,925 / 492 (16.8%) |
| Features | 64, built from the import readings only (`src/features.py`) |
| Target | `solar_installed`: 1 from the first day the export register shows more than 0.1 kWh, for homes that exported more than 1 kWh in total |
| Train / test split | 2,333 / 592 households, grouped by household |

The export readings define the target, so they, and everything tied to them (export power, reactive power,
power factor, the EXPORT flag and the meter type), are never used as model inputs.

Cleaning (`src/data.py`) turns the cumulative registers into 15-minute energy, removes meter resets and
impossible jumps, and keeps only days with at least 90 of 96 readings (no imputation).
