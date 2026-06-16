# AI-Driven Predictive Maintenance & RUL Forecasting

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![XGBoost](https://img.shields.io/badge/XGBoost-2B0203?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## 📌 Project Overview

This project uses the NASA C-MAPSS turbofan engine degradation dataset. This pipeline ingests raw, noisy multi-sensor telemetry to predict the **Remaining Useful Life (RUL)** of physical hardware.

By accurately predicting hardware failure weeks in advance, this framework enables manufacturing and procurement teams to move from reactive maintenance to proactive, data-driven spare parts ordering—minimizing stockouts, reducing factory downtime, and optimizing inventory carrying costs.

## Technical Methodology

To handle the highly non-linear degradation and environmental noise of industrial equipment, this pipeline utilizes several Data Engineering techniques:

1. **Piecewise Linear RUL Capping:** Prevents the model from over-penalizing healthy engines by capping the maximum RUL at 125 cycles, anchoring the model's focus strictly on the degradation phase.
2. **K-Means Environmental Normalization:** Deployed an unsupervised K-Means clustering algorithm to dynamically identify 6 distinct machine operating conditions (altitude, speed, temperature), normalizing sensor data to isolate pure mechanical wear from environmental noise.
3. **Time-Series Feature Engineering:** Engineered rolling means and rolling standard deviations over 5-cycle windows to smooth high-frequency sensor vibration and capture true degradation trajectories.
4. **XGBoost Architecture:** Trained a `XGBRegressor` to capture the complex, non-linear relationships of the engineered features.

## Results

**Performance Metrics:** The combination of Environmental Normalization and Time-Series Smoothing reduced the baseline prediction error by over **45%**. The final model achieved stable predictions across all four operating environments:

* **FD001 (Basic Environment):** RMSE ~ 17.8 cycles
* **FD002 (Multi-Condition):** RMSE ~ 16.4 cycles
* **FD003 (Multi-Fault):** RMSE ~ 19.6 cycles
* **FD004 (Complex/Noisy):** RMSE ~ 19.3 cycles

## How to Run

1. Clone the repository.
2. Download the C-MAPSS dataset from Kaggle and place the `.txt` files in the root directory.
3. Install dependencies: `pip install pandas numpy scikit-learn xgboost matplotlib`
4. Run the pipeline: `python cmapss_xgboost_trainer.py`
