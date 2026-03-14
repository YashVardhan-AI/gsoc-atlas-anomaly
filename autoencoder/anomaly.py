import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Configuration & Constants ---
TRAIN_FILES = [
    'train/data_none.txt', 
    'train/data_1.txt', 
    'train/data_2.txt', 
    'train/data_3.txt',
    'train/data_4.txt' 
]
TEST_FILE = 'test.txt'
WINDOW_SIZE = 15  
FEATURES = ['pss', 'vmem', 'nthreads', 'stime_rate']

# --- Data Processing Functions ---
def load_and_engineer_features(filepath):
    #load and engineer stime_rate feature
    df = pd.read_csv(filepath, sep=r'\s+')
    df['stime_rate'] = df['stime'].diff() / 2
    df.fillna(0, inplace=True)
    return df

def slide(data, window_size):
    # Converts raw data into sequences for LSTM input
    sequences = [data[i : i + window_size] for i in range(len(data) - window_size + 1)]
    return np.array(sequences)

def prepare_train_data(train_files, features, window_size):
    #preprocess data
    train_dfs = [load_and_engineer_features(f) for f in train_files]
    train_raw_list = [df[features].values for df in train_dfs]

    scaler = StandardScaler()
    all_train_raw = np.vstack(train_raw_list)
    scaler.fit(all_train_raw)

    X_train_list = []
    for raw_data in train_raw_list:
        scaled_data = scaler.transform(raw_data)
        sequences = slide(scaled_data, window_size)
        X_train_list.append(sequences)

    X_train = np.vstack(X_train_list)
    return X_train, scaler

def prepare_testing_data(test_file, features, window_size, scaler):

    df_test = load_and_engineer_features(test_file)
    test_raw = df_test[features].values
    test_scaled = scaler.transform(test_raw)
    X_test = slide(test_scaled, window_size)
    return X_test, df_test

# --- Model Functions ---
def build_autoencoder(window_size, num_features):
    #build autoencoder model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(WINDOW_SIZE, num_features), return_sequences=True),
        Dropout(0.2),
        LSTM(16, activation='relu', return_sequences=False),
        RepeatVector(window_size),
        LSTM(16, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(num_features))
    ])
    model.compile(optimizer='adam', loss='huber')
    return model

def train_model(model, X_train, epochs=60, batch_size=64):
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    return history

def save_model(model, filepath):
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    model = tf.keras.models.load_model(filepath)
    print(f"Model loaded from {filepath}")
    return model
# --- Anomaly Detection & Visualization Functions ---
def detect_anomalies(model, X_train, X_test):
    """Calculates the anomaly threshold from training data and detects anomalies in test data."""
    X_train_pred = model.predict(X_train, verbose=0)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=(1, 2))
    
    threshold = np.percentile(train_mae_loss, 99)
    print(f"Threshold set at MAE: {threshold:.4f}")

    X_test_pred = model.predict(X_test, verbose=0)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=(1, 2))

    anomalies = test_mae_loss > threshold
    print(f"Detected {np.sum(anomalies)} anomalous sequences.")
    
    return threshold, test_mae_loss, anomalies

def plot_results(df_test, test_mae_loss, threshold, window_size):
    """Generates the visualization for system usage vs. anomaly detection."""
    padding = np.full(window_size - 1, np.nan)
    aligned_loss = np.concatenate([padding, test_mae_loss])
    time_in_seconds = np.arange(len(df_test)) * 2

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_in_seconds, df_test['pss'], label='Physical RAM (PSS) in KB', color='blue')
    plt.title('System Memory Usage vs. Anomaly Detection')
    plt.ylabel('Memory (KB)')
    plt.legend(loc='upper left')

    plt.subplot(2, 1, 2)
    plt.plot(time_in_seconds, aligned_loss, label='Reconstruction Error (MAE)', color='purple')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold')

    plt.fill_between(
        time_in_seconds, 0, aligned_loss, 
        where=(np.nan_to_num(aligned_loss) > threshold), 
        color='red', alpha=0.3, label='Anomaly Detected'
    )

    plt.xlabel('Time (Seconds)')
    plt.ylabel('Error Margin')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---
def main():
    model_filepath = 'autoencoder_model.h5'

    X_train, scaler = prepare_train_data(TRAIN_FILES, FEATURES, WINDOW_SIZE)
    X_test, df_test = prepare_testing_data(TEST_FILE, FEATURES, WINDOW_SIZE, scaler)

    num_features = X_train.shape[2]
    
    if tf.io.gfile.exists(model_filepath):
        print(f"Loading existing model from {model_filepath}...")
        model = load_model(model_filepath)
    else:
        model = build_autoencoder(WINDOW_SIZE, num_features)
        train_model(model, X_train)
        save_model(model, model_filepath)

    threshold, test_mae_loss, _ = detect_anomalies(model, X_train, X_test)
    plot_results(df_test, test_mae_loss, threshold, WINDOW_SIZE)

if __name__ == "__main__":
    main()