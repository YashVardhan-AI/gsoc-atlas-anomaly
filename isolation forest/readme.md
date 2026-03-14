# Memory Anomaly Detection (Isolation Forest)

## Overview
This tool monitors program memory logs to automatically detect abnormal behavior, such as sudden spikes or gradual leaks. It focuses exclusively on analyzing a single system metric: **`pss` (Proportional Set Size)**, which represents the actual physical memory footprint of a process.

## Data Engineering: Adding Temporal Context
Standard anomaly detectors treat memory logs as isolated numbers, which makes them "time-blind" and unable to spot slow memory leaks. To solve this, the script engineers a 2D feature set from the 1D `pss` data: (this idea was given by AI)

* **Raw State:** The exact `pss` memory value at a specific point in time.
* **Rolling Trend (Velocity):** The calculated slope of the memory over a recent rolling window (e.g., the last 30 seconds). 

By pairing the raw value with its current trend, the model gains temporal context. It can instantly differentiate between a perfectly normal high-memory state (trend = 0) and an abnormal, creeping memory leak (trend > 0).

## How It Works: Isolation Forest
The engineered 2D data is fed into an **Isolation Forest**, an unsupervised machine learning algorithm. 

* **The Concept:** Anomalies in memory are "few and different." The algorithm works by randomly splitting the data to isolate individual points.
* **The Detection:** Rare and extreme memory patterns (like a massive spike or a steep leak trend) sit far away from normal data clusters, meaning they get isolated almost immediately. 
* **The Result:** Data points that require very few splits to isolate are automatically flagged as anomalies. Because the model is unsupervised, it requires no labeled training data and learns your program's unique baseline on the fly.

## Configuration & Tuning
The script's sensitivity can be tuned using two primary variables:

* **`WINDOW_SIZE`:** Controls the number of data points used to calculate the rolling trend. A larger window looks at broader, long-term behavior, while a smaller window reacts quickly to short-term changes.
* **`CONTAMINATION`:** Controls the strictness of the anomaly flagging. Lowering this value (e.g., `0.002`) forces the model to only flag the most extreme, severe outliers.

## Output
When executed, the script processes the logs and generates a `matplotlib` visualization. The normal `pss` memory curve is plotted in blue, with all detected anomalies distinctly overlaid as red dots for quick visual inspection.

##AI-use
other than the data engineering part ai was used in the plotting code and for helping write the readme file.
