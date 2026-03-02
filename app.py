import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="ANN Hyperparameter Tuning Dashboard", layout="wide")

st.title("❤️ Heart Disease ANN Hyperparameter Tuning Dashboard")

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Drop unnecessary columns safely
df = df.drop(columns=[col for col in df.columns if "Patient" in col], errors='ignore')
df = df.drop(columns=[col for col in df.columns if "Date" in col], errors='ignore')

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ===============================
# Preprocessing
# ===============================
X = df.drop("Heart Disease Status", axis=1)
y = df["Heart Disease Status"]

X = pd.get_dummies(X, drop_first=True)

if y.dtype == "object":
    y = y.map({"Yes": 1, "No": 0})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================
# Sidebar Hyperparameter Controls
# ===============================
st.sidebar.header("⚙️ Hyperparameter Tuning")

learning_rate = st.sidebar.selectbox(
    "Learning Rate",
    [0.1, 0.01, 0.001]
)

num_layers = st.sidebar.slider(
    "Number of Hidden Layers",
    min_value=1,
    max_value=3,
    value=1
)

neurons = st.sidebar.slider(
    "Neurons per Layer",
    min_value=16,
    max_value=128,
    value=64
)

dropout_rate = st.sidebar.slider(
    "Dropout Rate",
    min_value=0.0,
    max_value=0.5,
    value=0.2
)

epochs = st.sidebar.slider(
    "Epochs",
    min_value=10,
    max_value=100,
    value=20
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [16, 32, 64]
)

optimizer_name = st.sidebar.selectbox(
    "Optimizer",
    ["adam", "sgd", "rmsprop"]
)

# ===============================
# Model Builder
# ===============================
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(X_train.shape[1],)))

    for _ in range(num_layers):
        model.add(keras.layers.Dense(neurons, activation='relu'))
        model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    optimizer = keras.optimizers.get(optimizer_name)
    optimizer.learning_rate = learning_rate

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# ===============================
# Train Model Button
# ===============================
if st.button("🚀 Train ANN Model"):

    model = build_model()

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ Test Accuracy: {acc:.4f}")

    col1, col2 = st.columns(2)

    # Accuracy Curve
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(history.history['accuracy'], label="Train Accuracy")
        ax1.plot(history.history['val_accuracy'], label="Validation Accuracy")
        ax1.set_title("Accuracy Curve")
        ax1.legend()
        st.pyplot(fig1)

    # Loss Curve
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(history.history['loss'], label="Train Loss")
        ax2.plot(history.history['val_loss'], label="Validation Loss")
        ax2.set_title("Loss Curve")
        ax2.legend()
        st.pyplot(fig2)

    # Confusion Matrix
    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)
    