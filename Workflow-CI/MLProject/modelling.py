import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Bidirectional, Concatenate, Conv1D, Dense, Dropout,
    Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D,
    Input, LSTM, SpatialDropout1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

VOCAB_SIZE   = 20_000
MAX_LEN      = 100
EMBED_DIM    = 128
LSTM_UNITS   = 64
CONV_FILTERS = 64
CONV_KERNEL  = 3
DROPOUT_RATE = 0.3
BATCH_SIZE   = 256
EPOCHS       = 10
LABEL_MAP    = {"negatif": 0, "netral": 1, "positif": 2}
NUM_CLASSES  = 3


def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["clean_text", "label"]).copy()
    df["clean_text"] = df["clean_text"].astype(str)
    return df[df["clean_text"].str.len() > 0].copy()


def build_model(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, max_len=MAX_LEN,
                lstm_units=LSTM_UNITS, conv_filters=CONV_FILTERS,
                conv_kernel=CONV_KERNEL, dropout_rate=DROPOUT_RATE,
                num_classes=NUM_CLASSES):
    inp  = Input(shape=(max_len,))
    x    = Embedding(vocab_size, embed_dim, input_length=max_len)(inp)
    x    = SpatialDropout1D(dropout_rate)(x)
    conv = Conv1D(conv_filters, conv_kernel, activation="relu", padding="same")(x)
    c_max = GlobalMaxPooling1D()(conv)
    c_avg = GlobalAveragePooling1D()(conv)
    lstm  = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    l_max = GlobalMaxPooling1D()(lstm)
    l_avg = GlobalAveragePooling1D()(lstm)
    merged = Concatenate()([c_max, c_avg, l_max, l_avg])
    merged = Dropout(dropout_rate)(merged)
    merged = Dense(128, activation="relu")(merged)
    merged = Dropout(dropout_rate)(merged)
    out    = Dense(num_classes, activation="softmax")(merged)
    model  = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main(data_path, epochs=EPOCHS):
    print("Memuat data...")
    df = load_data(data_path)
    df["label_id"] = df["label"].map(LABEL_MAP)
    y = to_categorical(df["label_id"].values, num_classes=NUM_CLASSES)
    X = df["clean_text"].values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=df["label_id"].values)
    X_tr, X_test, y_tr, y_test = train_test_split(
        X_tr, y_tr, test_size=0.125, random_state=SEED)

    tok = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tok.fit_on_texts(X_tr)

    def to_seq(texts):
        return pad_sequences(tok.texts_to_sequences(texts),
                             maxlen=MAX_LEN, padding="post", truncating="post")

    X_tr_seq, X_val_seq, X_test_seq = to_seq(X_tr), to_seq(X_val), to_seq(X_test)

    y_tr_ids = np.argmax(y_tr, axis=1)
    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_tr_ids)
    cw_dict = {i: float(w) for i, w in enumerate(cw)}

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", ""))

    # Gunakan run ID yang sudah dibuat oleh MLProject
    run_id = os.environ.get("MLFLOW_RUN_ID")
    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_params({
            "vocab_size": VOCAB_SIZE,
            "max_len":    MAX_LEN,
            "embed_dim":  EMBED_DIM,
            "lstm_units": LSTM_UNITS,
            "batch_size": BATCH_SIZE,
        })

        model = build_model()
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ]
        model.fit(X_tr_seq, y_tr, validation_data=(X_val_seq, y_val),
                  epochs=epochs, batch_size=BATCH_SIZE,
                  class_weight=cw_dict, callbacks=callbacks, verbose=1)

        loss, acc = model.evaluate(X_test_seq, y_test, verbose=0)
        mlflow.log_metrics({"test_loss": loss, "test_accuracy": acc})
        print(f"Test accuracy: {acc:.4f}")

        # Simpan model sebagai artefak
        Path("model_output").mkdir(exist_ok=True)
        model.save("model_output/model.keras")
        tok_path = "model_output/tokenizer.json"
        with open(tok_path, "w") as f:
            json.dump(tok.to_json(), f)
        mlflow.log_artifact("model_output/model.keras")
        mlflow.log_artifact(tok_path)

        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
        default="dataset_sentimen_preprocessing.csv")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    main(args.data, args.epochs)
