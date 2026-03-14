import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def engineer_multivariate_features(df, memory_cols, window_size):
    """
    Takes multiple memory columns and returns a combined feature set containing
    both the raw values and the rolling trends for every column.
    """
    features = pd.DataFrame()
    
    def get_slope(y):
        x = np.arange(len(y))
        return np.round(np.polyfit(x, y, 1)[0], 3)
        
    for col in memory_cols:
        # 1. Add raw memory value
        features[f'{col}_raw'] = df[col]
        
        # 2. Add rolling trend
        rolling_slope = df[col].rolling(window=window_size).apply(get_slope, raw=True).fillna(0)
        features[f'{col}_trend'] = rolling_slope
        
    return features

def train_isolation_forest(df, memory_cols, window_size=30, contamination=0.02):
    #train
    X = engineer_multivariate_features(df, memory_cols, window_size)
    
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    
    model.fit(X.values)
    
    return model

def predict_anomalies(model, df, memory_cols, window_size=30):
    #predict
 
    X = engineer_multivariate_features(df, memory_cols, window_size)
    
    print("Running predictions on test data...")
    predictions = model.predict(X.values)
    anomaly_scores = model.decision_function(X.values)
    
    results_df = df.copy()
    results_df['anomaly_score'] = anomaly_scores
    results_df['is_anomaly'] = predictions == -1
    
    for col in memory_cols:
        results_df[f'{col}_trend'] = X[f'{col}_trend']
        
    return results_df

if __name__ == "__main__":
    TRAIN_FILE = 'data.txt'        
    TEST_FILE = 'test.txt'    
    
    TARGET_COLUMNS = ['pss'] 
    WINDOW_SIZE = 20 
    CONTAMINATION = 0.001

    # 1. Load datasets
    print(f"Loading training data from {TRAIN_FILE}...")
    train_data = pd.read_csv(TRAIN_FILE, sep=r'\s+')
    # Overwrite Time column: line number (index) * 2
    train_data['Time'] = train_data.index * 2
    
    print(f"Loading test data from {TEST_FILE}...")
    test_data = pd.read_csv(TEST_FILE, sep=r'\s+')
    # Overwrite Time column: line number (index) * 2

    test_data['Time'] = test_data.index * 2
    
    trained_model = train_isolation_forest(train_data, TARGET_COLUMNS, WINDOW_SIZE, CONTAMINATION)

    print("\n--- Starting Evaluation ---")
    results = predict_anomalies(trained_model, test_data, TARGET_COLUMNS, WINDOW_SIZE)
    
    anomalies = results[results['is_anomaly']]
    
    print("\n" + "="*40)
    print(f"Total TEST rows analyzed: {len(results)}")
    print(f"Total anomalies found in TEST data: {len(anomalies)}")
    print("="*40)
    
    # --- Visualization (Plotting the Test Data) ---
    print("\nGenerating subplots for test data visualization...")
    fig, axes = plt.subplots(len(TARGET_COLUMNS), 1, figsize=(14, 5 * len(TARGET_COLUMNS)), sharex=True)
    
    if len(TARGET_COLUMNS) == 1:
        axes = [axes]
        
    for i, col in enumerate(TARGET_COLUMNS):
        ax = axes[i]
        ax.plot(results['Time'], results[col], label=f'Normal {col.upper()}', color='blue', alpha=0.6)
        ax.scatter(anomalies['Time'], anomalies[col], color='red', label='Anomaly', zorder=5)
        ax.set_title(f'TEST DATA: {col.upper()} Usage')
        ax.set_ylabel('Memory (Bytes/KB)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.xlabel('Time (Seconds)')
    plt.tight_layout()
    plt.show()

