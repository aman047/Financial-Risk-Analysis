import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

SEQ_LENGTH = 60  # Sequence length (same as used in training)


# Function to preprocess training data
def prepare_data_for_training(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    joblib.dump(scaler, "scaler.pkl")  # Save scaler for later use

    X_train, y_train = [], []
    for i in range(SEQ_LENGTH, len(scaled_data)):
        X_train.append(scaled_data[i - SEQ_LENGTH:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train


# Function to preprocess test data
def prepare_data_for_testing(df, scaler):
    scaled_data = scaler.transform(df)

    X_test, y_test = [], []
    for i in range(SEQ_LENGTH, len(scaled_data)):
        X_test.append(scaled_data[i - SEQ_LENGTH:i, 0])
        y_test.append(scaled_data[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_test, y_test
