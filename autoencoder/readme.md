# LSTM Autoencoder for Multivariate Memory Anomaly Detection

## Project Overview
This project implements a **Deep Funnel LSTM Autoencoder** designed to detect complex memory anomalies in Linux systems. Unlike simple threshold-based monitors, this system uses Deep Learning to understand the mathematical relationships between multiple system metrics simultaneously.

The model is trained exclusively on "Normal" system behavior. Once trained, it can detect anomalies—such as **memory leaks**, **fork-bombs**, or **unauthorized massive allocations**—by identifying patterns that deviate from the learned baseline.

Data is captured using **[prmon](https://github.com/HSF/prmon)** at a 2-second polling interval.

---

## Monitored Features
To ensure high sensitivity to memory-related issues while ignoring background "noise," the model monitors **4 key features**:

| Feature | Description | Why it's used |
| :--- | :--- | :--- |
| **`pss`** | Proportional Set Size | Ground truth for physical RAM footprint. |
| **`vmem`** | Virtual Memory | Early warning signal for massive allocation requests. |
| **`nthreads`** | Thread Count | Distinguishes heavy load from a fork-bomb anomaly. |
| **`stime_rate`** | System CPU Rate | Tracks kernel overhead of managing memory pages. |

---

## Model Architecture
The architecture is a **Deep Funnel Autoencoder**. It forces the system data through a narrow "bottleneck" to ensure it learns underlying rules rather than memorizing data points.

* **Encoder:** Two LSTM layers (64 → 16 units) compress 30 seconds of history (15 time-steps).
* **Bottleneck:** A 16-unit vector holding the "Normal" system state.
* **Decoder:** Two LSTM layers (16 → 64 units) reconstruct the original window.
* **Regularization:** Dropout layers prevent overfitting to minor system noise.

---

## How Anomaly Detection Works
The detector operates on the principle of **Reconstruction Error**:

1.  **Compression:** The model attempts to compress and then rebuild the incoming data.
2.  **The Mistake (MAE):** Since the model only knows "Normal" data, it will fail to accurately rebuild anomalous patterns (like a leak). The difference between the real data and the rebuild is the **Mean Absolute Error (MAE)**.
3.  **The Threshold:** We calculate the **99th percentile** of the error found during training. This is our "Alarm Line."
4.  **Flagging:** Any sequence with a reconstruction error higher than this threshold is flagged as an anomaly.

---

## Data Structure
The pipeline expects a `train/` directory for baselines and a root-level test file for evaluation.

```text
├── train/
│   ├── data_none.txt   # State: Idle baseline
│   ├── data_1.txt      # State: Light background allocation
│   ├── data_2.txt      # State: Medium steady allocation
│   ├── data_3.txt      # State: Heavy steady allocation
│   └── data_4.txt      # State: Back to low
├── test.txt      # Evaluation data (Normal + Injected Anomalies)
├── anomaly.py # The Python/TensorFlow pipeline for traing and inference
```

## AI-usage 
Ai was used for writing the plotting code and copilot was on while writing the code also the idea of dynamic threshhold was given by ai. It was also used in writting the readme.