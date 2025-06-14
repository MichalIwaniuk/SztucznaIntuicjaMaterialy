# train.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import librosa
import numpy as np
import argparse

INSTRUMENTS = []

def load_audio_file(file_path, sr=22050, duration=5):
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        if len(y) < sr * duration:
            y = np.pad(y, (0, sr * duration - len(y)))
        return y
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return None

def compute_melspectrogram(y, sr=22050, n_mels=128):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

def load_train_data(train_dir):
    global INSTRUMENTS
    X, Y = [], []

    for instrument in sorted(os.listdir(train_dir)):
        instrument_dir = os.path.join(train_dir, instrument)
        if not os.path.isdir(instrument_dir):
            continue
        for fname in os.listdir(instrument_dir):
            if not fname.lower().endswith('.wav'):
                continue
            y = load_audio_file(os.path.join(train_dir, instrument, fname))
            if y is None:
                continue
            mel = compute_melspectrogram(y)
            mel = mel[..., np.newaxis]  # (time, freq, 1)
            X.append(mel)
            Y.append([instrument])

    INSTRUMENTS = sorted({inst for sub in Y for inst in sub})
    mlb = MultiLabelBinarizer(classes=INSTRUMENTS)
    Y_bin = mlb.fit_transform(Y)

    X = np.array(X)
    Y = np.array(Y_bin)

    print(f"[INFO] Załadowano {len(X)} plików dźwiękowych z {len(INSTRUMENTS)} klasami")
    return X, Y

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)  # sigmoid dla multi-label
    return Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data/train')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--model_out', default='models/model.h5')
    args = parser.parse_args()

    tf.get_logger().setLevel('ERROR')

    # Wczytanie danych
    X, Y = load_train_data(args.train_dir)

    # Ręczny podział na zbiory
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )

    # Przygotowanie tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)) \
        .shuffle(1000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)) \
        .batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # Budowa i kompilacja modelu
    input_shape = X_train.shape[1:]  # np. (time, freq, 1)
    num_classes = Y.shape[1]
    model = build_model(input_shape, num_classes)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Checkpoint i EarlyStopping z patience=5 na val_loss
    checkpoint = ModelCheckpoint(
        args.model_out, save_best_only=True, monitor='accuracy', mode='min'
    )
    early = EarlyStopping(
        monitor='accuracy',
        patience=10000,
        restore_best_weights=False,
        mode='min',
        verbose=1
    )

    # Trening
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint, early],
        verbose=1
    )

    # Zapis finalnego modelu
    model.save(args.model_out)
    print(f"[INFO] Model zapisany do {args.model_out}")
